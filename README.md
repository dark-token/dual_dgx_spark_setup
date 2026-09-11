# Dual NVIDIA DGX Spark Cluster: Setup Guide, NCCL Tuning & Pitfalls

A field-tested guide for connecting two DGX Sparks (GB10 Grace Blackwell) into a working
inference cluster: from first boot, through a 200GbE QSFP fabric tuned from **2.68 → 21.98 GB/s
NCCL bus bandwidth**, to serving **DeepSeek-V4-Flash (284B MoE)** with vLLM across both nodes.

Everything here was learned the hard way — including setting the units up from a hotel room
with no monitor, keyboard, or Ethernet cable. The pitfalls section is the part I wish had
existed before I started.

> Hardware: 2× DGX Spark (128 GB unified memory each, 4 TB NVMe, ConnectX-7 2×QSFP 200GbE)
> connected with one QSFP56 passive DAC cable, port 0 ↔ port 0. DGX OS 7.x (Ubuntu 24, aarch64).

---

## Table of Contents

1. [First boot (headless)](#1-first-boot-headless)
2. [Wi-Fi / network pitfalls](#2-wi-fi--network-pitfalls)
3. [QSFP fabric setup](#3-qsfp-fabric-setup)
4. [SSH mesh](#4-ssh-mesh)
5. [Enabling & tuning NCCL (2.68 → 22 GB/s)](#5-enabling--tuning-nccl)
6. [Serving big models with vLLM (spark-vllm-docker)](#6-serving-big-models-with-vllm)
7. [Full pitfall catalogue](#7-full-pitfall-catalogue)
8. [Useful aliases & a jumpstart script](#8-useful-aliases--jumpstart)

---

## 1. First boot (headless)

Each Spark broadcasts a **setup Wi-Fi hotspot on first boot only** (SSID + password are on a
sticker on the Quick Start card). Join it from a laptop; a captive-portal setup page opens.

Rules that will save you pain:

- **Use the same username on both units.** MPI/rsync/cluster scripts assume it.
- **Save the password immediately.** There is no default and no remote reset — recovering a
  lost password requires a monitor + keyboard (GRUB single-user mode).
- Plug wired Ethernet in **before** powering on if you have it; it's the most reliable path.
- The device starts installing updates after joining your network. **Do not cut power** —
  this phase takes ~10 minutes with reboots.
- Once setup completes, **the hotspot never comes back**. A configured Spark is a silent
  network client. All later access is SSH (`ssh user@spark-xxxx.local` — mDNS works out of
  the box on macOS) or the DGX Dashboard web UI.

### If you're stuck with no monitor/keyboard (the hotel scenario)

Do **not** set the Spark up against hotel/airport Wi-Fi: captive portals block its internet,
and client isolation makes it unreachable from your laptop even when connected. Options that
actually work:

- **Phone hotspot** (Maximize Compatibility on) for the setup wizard — no portal, no isolation.
- **USB-C Ethernet adapter + patch cable** (≈ $20) — the universal rescue: share your laptop's
  internet (macOS: Internet Sharing from *iPhone USB* → *Ethernet adapter*), cable straight
  into the Spark's RJ-45. Your laptop becomes its DHCP server; `ping spark-xxxx.local`,
  SSH in, fix the Wi-Fi with `nmcli`. Keep this adapter forever — it's your break-glass access.
- macOS Internet Sharing **cannot broadcast an open (password-less) Wi-Fi network**, so you
  can't spoof an open hotel SSID from a Mac. A home router can (clone the SSID as open,
  isolation off) if the units got bound to a network you can't reach.

```bash
# once you're in, bind the unit to networks you control:
sudo nmcli device wifi connect "YourSSID" password "yourpassword"
sudo nmcli connection add type wifi con-name home ssid "HomeSSID" \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "homepassword"
```

---

## 2. Wi-Fi / network pitfalls

- **`ping` resolves but `No route to host`** → client isolation (hotels, guest SSIDs, some
  mesh defaults). The devices are on the network; the AP blocks peer traffic.
- **mDNS names going stale** after IP changes → trust `ping <name>.local` output, verify
  against the router's client list, fall back to raw IPs.
- **Dual-homing (Wi-Fi + Ethernet simultaneously)** causes confusing asymmetric behavior.
  Pick one management path per unit. Disable Wi-Fi once wired: `sudo nmcli radio wifi off`.
- **Reserve DHCP addresses** for both units in your router (per interface — wired MACs are
  different MACs!), and put hostnames in `~/.ssh/config` on your laptop.
- macOS redacts SSIDs in Terminal output (`<redacted>`) unless Terminal has Location
  Services permission. The Wi-Fi menu always shows the real name.

---

## 3. QSFP fabric setup

One passive DAC cable, **same physical port on both units** (port 0 ↔ port 0). No switch
needed for two nodes. Each physical port shows up as **two logical interfaces / RDMA
devices** on GB10 (two PCIe halves of the ConnectX-7):

| netdev | RDMA device |
|---|---|
| `enp1s0f0np0` | `rocep1s0f0` |
| `enP2p1s0f0np0` | `roceP2p1s0f0` |

**Configure BOTH halves, on BOTH nodes, symmetrically** — asymmetry here caused my worst
NCCL crash (see pitfalls). Static IPs on distinct subnets + jumbo frames:

```bash
# node 1 (head)
sudo nmcli con add type ethernet ifname enp1s0f0np0   con-name qsfp0  \
  ipv4.method manual ipv4.addresses 192.168.177.11/24 802-3-ethernet.mtu 9000
sudo nmcli con add type ethernet ifname enP2p1s0f0np0 con-name qsfp0b \
  ipv4.method manual ipv4.addresses 192.168.178.11/24 802-3-ethernet.mtu 9000
sudo nmcli con up qsfp0 && sudo nmcli con up qsfp0b

# node 2 (worker): same, with .12 addresses
```

Why static instead of the link-local (169.254.x.x) that NVIDIA's playbook defaults to:
link-local puts both interfaces in one /16, which breaks subnet-based autodiscovery in
cluster tooling (spark-vllm-docker refuses with *"interfaces share the same subnet"*), and
static IPs are simply deterministic.

Verify:

```bash
ibdev2netdev                          # interfaces show (Up)
ethtool enp1s0f0np0 | grep Speed      # must be 200000Mb/s
ping -c 3 -M do -s 8972 -I enp1s0f0np0 192.168.177.12   # jumbo frames end-to-end
```

Cleanup traps:

```bash
nmcli con show                        # look for DUPLICATE qsfp profiles → delete inactive twins by UUID
sudo nmcli con delete "Wired connection 1"   # kill auto-created DHCP profiles stuck on
                                             # "connecting (getting IP configuration)" —
                                             # there's no DHCP server on a p2p cable
```

Raw RDMA sanity check (per-QP ceiling on GB10 is ~1.6 GB/s; ~13.3 GB/s per NIC half with 8 QPs):

```bash
sudo apt install -y perftest
ib_write_bw -d rocep1s0f0 -q 8                       # node 1 (server)
ib_write_bw -d rocep1s0f0 -q 8 192.168.177.11        # node 2 (client)
```

---

## 4. SSH mesh

Passwordless SSH in **both directions** between the nodes (and from your laptop) is required
by every cluster tool (MPI, Ray, spark-vllm-docker autodiscovery, hf-download distribution):

```bash
# on each node:
ssh-keygen -t ed25519          # defaults, empty passphrase
ssh-copy-id user@<other-node-ip>
ssh user@<other-node-ip> hostname   # must return the peer hostname with no prompt
```

Accept fingerprints once for every address you'll use (LAN IPs *and* QSFP IPs).

---

## 5. Enabling & tuning NCCL

### Build (both nodes)

NCCL must be compiled for Blackwell GB10 (`sm_121`):

```bash
sudo apt install -y build-essential openmpi-bin libopenmpi-dev git
git clone https://github.com/NVIDIA/nccl.git && cd nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

cd ~ && git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi NCCL_HOME=$HOME/nccl/build -j

echo 'export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Find YOUR GID index (do not copy someone else's)

NCCL over RoCE needs the GID index of the **RoCE v2 + IPv4** entry. Mine was **5**; guides
often say 3. Wrong index = cryptic failures.

```bash
for i in 0 1 2 3 4 5 6 7; do
  echo "index $i: type=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null) \
gid=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null)"
done
# pick the index where type = "RoCE v2" AND the gid embeds your IPv4 (::ffff:a9fe:... / ::ffff:c0a8:...)
```

### The winning environment (≈ 22 GB/s all_gather busbw)

Save as `~/nccl-cluster-env.sh` on both nodes (credit: the tested config in
[ArgentAIOS/dgx-spark-cluster](https://github.com/ArgentAIOS/dgx-spark-cluster), adapted —
device names and GID index differ per system):

```bash
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export OMPI_MCA_btl_tcp_if_include=enp1s0f0np0
export UCX_NET_DEVICES=rocep1s0f0:1
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0   # BOTH NIC halves — this is the 13→22 GB/s jump
export NCCL_IB_GID_INDEX=5                   # YOUR index from the check above
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4          # per-QP cap is ~1.6 GB/s; parallelism is everything
export NCCL_IB_TC=106
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NET_GDR_LEVEL=0                  # classic GPUDirect RDMA is NOT supported on GB10
export NCCL_DMABUF_ENABLE=0                  # unified memory; GDR assumptions cause hangs
```

### The test (run from the head node only — mpirun launches the peer via SSH)

```bash
source ~/nccl-cluster-env.sh
mpirun -np 2 -H 192.168.177.11:1,192.168.177.12:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME -x OMPI_MCA_btl_tcp_if_include -x UCX_NET_DEVICES \
  -x NCCL_IB_DISABLE -x NCCL_NET -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
  -x NCCL_IB_TIMEOUT -x NCCL_IB_RETRY_CNT -x NCCL_IB_QPS_PER_CONNECTION \
  -x NCCL_IB_TC -x NCCL_ALGO -x NCCL_PROTO \
  -x NCCL_NET_GDR_LEVEL -x NCCL_DMABUF_ENABLE \
  $HOME/nccl-tests/build/all_gather_perf -b 1G -e 4G -f 2
```

### My progression (all_gather busbw, 1–4 GB messages)

| Config | busbw |
|---|---|
| Defaults (one link-local iface, TCP-ish behavior) | **2.68 GB/s** |
| + second NIC half configured, jumbo frames | 2.99 GB/s |
| + correct GID index, QPS=4, Ring/Simple, GDR off (single HCA) | **13.33 GB/s** (= exactly the single-half ib_write_bw ceiling) |
| + `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` (both halves) | **21.98 GB/s avg / 23.07 peak** |

`iperf3` numbers (TCP) will look terrible (~13–16 Gb/s) regardless — that's CPU-bound TCP,
not your fabric. Judge the fabric by `ib_write_bw` and NCCL, not iperf.

---

## 6. Serving big models with vLLM

**Skip NVIDIA's NGC `run_cluster.sh` playbook path** — the `nvcr.io/nvidia/vllm` NGC image
I tested didn't even ship the `ray` CLI, and DeepSeek-V4-Flash needs kernels (DeepGEMM
`nv_dev`, B12X) that stock builds lack. Use the community project built for exactly this:

**[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — Spark-patched vLLM,
autodiscovery, no-Ray multi-node mode (default), model recipes.

```bash
git clone https://github.com/eugr/spark-vllm-docker.git && cd spark-vllm-docker
./build-and-copy.sh -c        # pulls tested image, auto-copies to the worker over QSFP
                               # (this is why static IPs + SSH mesh matter)

# download a model once, distribute over the 200GbE link:
./hf-download.sh deepseek-ai/DeepSeek-V4-Flash-0731 -c --copy-parallel   # needs uv/uvx installed

# serve with the tested recipe (head node only; everything runs from the head):
./run-recipe.sh deepseek-v4-flash-0731 --no-ray
```

Recipes encode the non-obvious parts: B12X backends need the **B12X image variant**
(`vllm-node-b12x` — recipes build it automatically), an environment constellation
(`VLLM_USE_B12X_*`, `CUTE_DSL_ARCH=sm_121a`, AOT compile flags), InstantTensor loading, sane
CUDA-graph budgets, and DSpark speculative decoding config. Hand-assembling these as CLI
flags is how I produced three of my five crashes.

Memory-safety flags worth adding on 128 GB unified-memory nodes:

```bash
./launch-cluster.sh --earlyoom --non-privileged --mem-limit-gb 105 exec vllm serve ...
```

Under memory pressure a Spark can starve `sshd` on **both nodes at once** (you lose all
remote access while the load either recovers or dies). `--earlyoom` + a container memory
limit converts "frozen hosts, power-button time" into "container OOM-killed, hosts fine".
Also useful: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` un-sticks stalled
loaders; keep an SSH session pre-opened on each node with that command ready before big
loads (new SSH connections die under pressure; existing ones usually survive).

Loader guidance: `--load-format fastsafetensors` is fast but memory-hungry — the README
warns against it for models taking > 0.85 of RAM, and a 157 GB model on 2×128 GB is over
that line (it froze both my hosts and ended in an NCCL broadcast timeout).
`--load-format instanttensor` loaded the same 157 GB in **21 seconds** without drama.

---

## 7. Full pitfall catalogue

Network & access:
1. **Hotel/captive-portal Wi-Fi**: Spark joins but has no internet (no browser to accept
   terms) and client isolation blocks your laptop from reaching it. Never set up against one.
2. **Configured Sparks never re-broadcast their hotspot.** No monitor + unknown network =
   locked out. Escape hatches: USB-C Ethernet adapter + Internet Sharing; router-side SSID
   clone at home; monitor + keyboard.
3. **iPhone hotspots always require a password**, so they can't impersonate an *open* SSID.
   Modern macOS Internet Sharing also refuses Security: None.
4. **Lost login password**: no remote reset exists. GRUB single-user mode (`Esc` at boot,
   `init=/bin/bash`, `mount -o remount,rw /`, `passwd user`) — requires physical keyboard.
5. **DGX OS first-boot wizard sometimes shows an empty Wi-Fi list.** Reboot into the wizard
   or plug Ethernet (a LAN port on any router works; the wizard prefers wired anyway).
6. **A mouse alone is enough** for GUI rescue: GNOME's on-screen accessibility keyboard at
   the login screen + a TV over HDMI.

QSFP / NCCL:
7. **`(incomplete)` ARP entries** = nobody answering on the wire; it's physical/boot-order,
   not software. Reboot with the cable attached.
8. **Auto-created "Wired connection N" profiles run DHCP forever** on the p2p link
   ("connecting (getting IP configuration)"). Replace with static/link-local profiles.
9. **Duplicate nmcli profiles with the same name** race at boot and eat your `con mod`
   edits. `nmcli con show`, delete twins by UUID.
10. **Configuring the second RoCE half on only ONE node** → GID type mismatch
    (`ibv_modify_qp failed with 22 … local GID ::ffff:IPv4, remote GID fe80::…`) → NCCL
    "unhandled system error" at init. Symmetry or nothing.
11. **Wrong `NCCL_IB_GID_INDEX`** and **assuming GPUDirect/peermem works on GB10** (it
    doesn't; DMA-BUF or CPU-staged only) are the two classic silent killers.
12. **Per-QP throughput caps at ~1.6 GB/s** on this NIC; without `QPS_PER_CONNECTION` you
    will conclude your cable is broken. It isn't.
13. Exported env vars **die with the terminal session** — mpirun `-x` forwards only what
    exists. Keep them in a sourceable file.
14. `Authorization required, but no authorization protocol specified` in mpirun output is
    X11 noise — harmless.

Cluster software:
15. **NGC vLLM image had no `ray` CLI** → `run_cluster.sh` containers exited instantly,
    leaving empty `docker ps` and no logs. Interrogate images before trusting playbooks:
    `docker run --rm IMAGE bash -ilc 'which ray; which vllm'`.
16. **vLLM's `run_cluster.sh` moved** in the repo (`examples/online_serving` →
    `examples/ray_serving`); wget 404s deceive.
17. **Stale `vllm_node` container on one node** makes the launcher think the cluster is up
    → "No such container" on the other. `docker rm -f vllm_node` on **both** before launch.
18. **Orchestration commands run ONCE, on the head node** (mpirun, launch-cluster,
    run-recipe, hf-download). Running them on both nodes creates dueling clusters. Only
    system config (nmcli, apt, reboots, cache drops) is per-node.
19. **tmux sessions live per-machine** — "can't find session" usually means you're SSH'd
    into the wrong Spark. `hostname` first. Also: detach (`Ctrl+B, D`) before attaching
    elsewhere; `tmux capture-pane -t NAME -p -S -3000 > file` beats scrollback archaeology.
20. **Checkpoint revisions matter enormously.** My `nvidia/...-NVFP4` quant was pre-0731:
    no DSpark draft module (`dspark_noise_token_id` error) and a `swiglu_limit` the B12X
    MoE kernel rejects. Recipes are tested against *specific* checkpoints — use exactly the
    one they name, or expect flag surgery per mismatch.
21. **FlashInfer sparse-MLA crashed during CUDA graph capture** on sm121
    (`sparse_mla_sm120.cu: eidx must be contiguous`) with default backends — the recipes'
    B12X attention backend exists to route around it, and it only registers in the B12X image.

---

## 8. Useful aliases & jumpstart

```bash
# ~/.zshrc on both nodes
alias dropcaches='sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
alias cleanup='docker rm -f vllm_node 2>/dev/null; docker ps -a'
alias vlogs='docker logs vllm_node 2>&1 | tail -30'
alias sparkcheck='docker ps -a; nvidia-smi | head -12; free -h; ip -br addr show enp1s0f0np0; ip -br addr show enP2p1s0f0np0'

# head node only
alias vstop='cd ~/spark-vllm-docker && ./launch-cluster.sh stop'
```

Post-power-cycle sequence: `sparkcheck` + `cleanup` on both → QSFP pings → `dropcaches`
on both → `run-recipe.sh <recipe> --no-ray` in tmux on the head → watch for
`Application startup complete`, then:

```bash
curl http://<head-lan-ip>:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

---

## Credits

- [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — cabling,
  discovery scripts, NCCL build reference
- [ArgentAIOS/dgx-spark-cluster](https://github.com/ArgentAIOS/dgx-spark-cluster) — the
  NCCL env-var set that took my fabric from 3 to 22 GB/s
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — the serving stack,
  recipes, and Spark-specific patches that made a 284B model run at all

*Written after ~a week of real debugging across a hotel room and a home office. May your
shard counters always advance.*
