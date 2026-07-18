# FRITZ!Box Delayed-Answer PoC

This proof of concept replaces Asterisk only for SIP signalling. A headless PJSUA2 client
registers as an internal FRITZ!Box 7690 IP telephone, sends `180 Ringing`, waits approximately
four rings, and answers only if no other handset accepted the call. It does not configure an
external SIP forwarding target.

The PoC intentionally has no OpenAI or audio bridge yet. After answering it stays silent and
hangs up after `POC_MAX_CALL_SECONDS`. This isolates the load-bearing assumption: when another
FRITZ!Box handset answers, the pending IP-phone leg must be cancelled before its timer expires.

## FRITZ!Box 7690 Setup

1. Open **Telefonie → Telefoniegeräte → Neues Gerät einrichten**.
2. Choose a telephone connected through **LAN/WLAN (IP-Telefon)**.
3. Name it `AI Answering Machine` and create a unique username and strong password.
4. Assign the same incoming number as the normal handsets. Do not enable external registration.
5. Keep the username and password for `.env.pjsip-poc`; do not commit them.

The Docker host must be on the FRITZ!Box LAN. The Compose service uses host networking because
SIP embeds addresses in its signalling and RTP later needs dynamically negotiated UDP ports.

## Run

```bash
cp .env.pjsip-poc.example .env.pjsip-poc
# Fill FRITZBOX_SIP_USERNAME and FRITZBOX_SIP_PASSWORD.
docker compose -f compose.pjsip-poc.yml build
docker compose -f compose.pjsip-poc.yml up
```

A successful registration produces a log similar to:

```text
SIP registration active=True status=200 OK
```

If the log only repeats outgoing `REGISTER` messages and never shows an incoming `401
Unauthorized`, the FRITZ!Box is not starting SIP authentication. Verify under **Telefonie →
Telefoniegeräte** that the device was created specifically as **LAN/WLAN (IP-Telefon)**. In its
login settings, re-enter the exact username and password used in `.env.pjsip-poc`; both must be
at least eight characters and the password must differ clearly from the username. The Docker
host must be in the FRITZ!Box home network, not guest access. If the timeout remains, AVM
recommends deleting the IP telephone, restarting the FRITZ!Box, and creating the device again.
Keep `PJSIP_TRANSPORT=udp`; the FRITZ!Box rejects the tested local SIP registration over TCP.

## Live Test Matrix

1. Call the assigned public number and answer on a normal handset before four rings. Expect a
   `487 Request Terminated`/disconnected log and no `answered by PoC agent` log.
2. Call again without answering. Expect `answered by PoC agent` after about 20 seconds.
3. Answer very close to the deadline repeatedly. Exactly one side must win; the container must
   remain registered and accept the next call.
4. Restart the container and verify automatic re-registration.

Tune `ANSWER_DELAY_SECONDS` until the real handset cadence yields four complete rings. Stop the
PoC with `Ctrl-C` or `docker compose -f compose.pjsip-poc.yml down`.

## Exit Criteria

- Registration remains active for at least 24 hours.
- Human pickup always cancels the pending agent leg.
- Unanswered calls are accepted at the configured deadline.
- Near-deadline races do not produce stuck or double-answered calls.
- Container restart does not affect normal FRITZ!Box telephony.

PJSIP is GPL-licensed unless used under a separate commercial license. Review distribution
requirements before publishing an image outside this private deployment.
