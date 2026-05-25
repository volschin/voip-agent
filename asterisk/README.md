# Asterisk Configuration

Copy these files to `/etc/asterisk/` on the NUC running Asterisk 20.

1. Replace all `<PLACEHOLDER>` values with real Fritzbox credentials.
2. Reload Asterisk: `asterisk -rx "core reload"`
3. Check registration: `asterisk -rx "pjsip show registrations"`
   Expected: `fritzbox` shows `Registered`.
4. Check ARI: `curl -u voip-agent:changeme http://localhost:8088/ari/applications`
   Expected: JSON list including `voip-agent`.
