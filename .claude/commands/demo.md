---
description: Prepare data and baseline if missing, launch the Streamlit demo, and walk through it
---
1. Run `make baseline` only if `data/test.csv` or `models/tfidf.joblib` is missing.
2. Start the app with the `streamlit` configuration in `.claude/launch.json` (preview_start), not with Bash.
3. Open each page (Home, Route a ticket, Compare models, Cost explorer, Error review) and confirm it renders without errors.
4. On "Route a ticket", run the "Angry cancellation" sample and report the intent, urgency, team, and whether it escalated.
5. Reply with the URL and a 3-bullet walkthrough a PM could follow in 5 minutes.
