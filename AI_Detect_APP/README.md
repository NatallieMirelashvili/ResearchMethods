# AI_Defect — Static GitHub Pages + Google Sheets (no server)

Upload these files to a GitHub repo and enable **GitHub Pages**. All answers are written to your Google Sheet via **Apps Script Web App**.

## Steps
1) In Google Sheets create a spreadsheet. Copy its ID from the URL.
2) In Apps Script paste `apps_script_webhook.js`, set `SHEET_ID` and `SECRET`, Deploy as **Web app** (Anyone). Copy the Web App URL.
3) In `config.json` set `webhook_url` to that URL, and `webhook_secret` to the same secret string.
4) Put your 20 images under `images/real/01.jpg..20.jpg` and `images/ai/01.jpg..20.jpg` (or use absolute URLs by editing `pairs.json`).
5) Commit & push to GitHub. Enable **Settings → Pages** for the repo. Share the link.

Each answer appends a row:
```
timestamp | participant_email | pair_id | left_url | right_url | left_type | right_type | choice | result
```
`result` is 'T' for correct, 'F' for incorrect.
