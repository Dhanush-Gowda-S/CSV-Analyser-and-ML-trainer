
# Spark ML Dashboard — Flask + PySpark

**A self-contained web app for exploratory data analysis, visualization, and Spark ML model training.**

---

## Contents

- `app.py` — Main Flask application (multi-part: upload/analyze/visualize/train/admin)
- `templates/` — Jinja2 HTML templates (dashboard, analyze, visualize, admin, profile, etc.)
- `static/` — CSS, JS, images (visualize.js, train_status.js, style.css)
- `models/` — Saved Spark models (created at training time)
- `static/uploads/` — Uploaded CSV files
- `app.db` — SQLite database for users, files, models, logs

---

## Features

- Upload CSV files and run automatic EDA (schema, missing values, correlations).
- Advanced client-side visualization using Chart.js with plugins:
  - Bar, Grouped Bar, Line, Scatter, Pie, Doughnut, Histogram, Boxplot (plugin), Heatmap (matrix plugin).
  - Auto-detection of numeric / categorical columns.
  - Download charts as PNG.
- Two ML modes:
  - **Option A (Auto All)** — uses all numeric features (optionally encodes categorical), automatic split and model selection.
  - **Option B (Correlation-based)** — selects features with correlation ≥ 0.20 to target.
- Model choices for regression/classification (Linear, RandomForest, GBT, Logistic, etc.).
- Background training with progress polling; trained model saved and downloadable as a ZIP.
- Admin panel:
  - Create new admins, promote/demote users, delete users.
  - Activity logs.
- User profiles with avatar upload; default avatar fallback.

---

## Quickstart (Development)

> Tested with Python 3.10+ and Spark 3.x locally. Use a virtual environment.

1. Clone repository (or copy files) into a project folder:
   ```bash
   git clone <repo> my-sparkml-app
   cd my-sparkml-app
   ```

2. Create and activate virtualenv:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt` should include:
   ```
   flask
   flask-login
   pyspark
   pandas
   Werkzeug
   ```

4. Start the app:
   ```bash
   python app.py
   ```
   App runs at `http://127.0.0.1:5000`.

5. Default admin account (auto-created on first run):
   - Username: `admin`
   - Password: `adminpass`
   > **Change immediately in production.**

---

## Upload & Analyze CSV

1. Login.
2. Dashboard → Analyze → Upload a CSV (`.csv` only).
3. The app runs a Spark-based EDA:
   - Cleaned column names
   - Column types (numeric vs categorical)
   - Missing value counts
   - Correlation preview for the top numeric columns
   - A sample JSON (first 2,000 rows) is passed to the visualization page.

Notes:
- The app sanitizes column names by replacing spaces, slashes, parentheses, and hyphens with underscores.
- Columns with all-null or constant values are dropped automatically.

---

## Visualization

- Visualize page uses a client-side JS (`visualize.js`) that:
  - Detects numeric/categorical columns using a sample of the dataset.
  - Enables chart types appropriate for selected columns.
  - Auto-hides plugin-dependent charts if plugins aren't available in the browser.
  - Allows downloading charts as PNG.

If a future CSV fails to render charts:
- Open Developer Console in your browser (Edge/Chrome) to see JS errors.
- Large CSVs may be heavy — the view uses a sample (first 5k rows) to keep the browser responsive.
- If plugin not available, boxplot/heatmap renders with fallback charts or warns.

---

## Training Models

- Go to the file Analyze view, choose **Train Model**:
  - Select `Target Column`.
  - Choose **Model Mode**:
    - **A** (Auto All features)
    - **B** (Correlation-based — selects features with |corr| ≥ 0.20)
    - **Manual** (supply selected features)
  - Pick `Model Type` (Regression/Classification) and algorithm.
  - Click **Start Training**.

Behavior:
- Training runs in a background thread and returns a `task_id`.
- Frontend polls `/train_status/<task_id>` to update the progress bar.
- When complete, a modal displays evaluation metrics and feature importance (if available).
- Trained models are saved under `models/` and can be downloaded as a ZIP.

Common issues:
- If progress reaches 100% but no result modal appears, check `/train_status/<task_id>` in Network tab or call the endpoint to see final JSON. The saved model ZIP appears in Dashboard → Models.

---

## Admin Panel

- Admins can:
  - Create new admin accounts.
  - Promote or demote existing users.
  - Delete users (removes related files/models/logs).
  - View recent activity logs.

Security notes:
- Only admin users have access to `/admin` endpoints.
- Do not use debug server in production; use Gunicorn or another WSGI server behind a reverse proxy.

---

## File/Model Storage

- Uploaded CSVs: `static/uploads/`
- User profile images: `static/profile/`
- Saved Spark models: `models/` — each model saved as SparkML pipeline directory, zipped on download.

Model ZIP contains the Spark saved pipeline directory structure (metadata, data of stages, model files). The exact files vary depending on the estimator used (RandomForest, GBT, LinearRegression, etc.). You can reload the model in Spark using `PipelineModel.load(path)`.

---

## Troubleshooting

- **Charts not showing**: Open browser console. Look for plugin loading errors or `Cannot set properties of null` (means the template element IDs changed).
- **Training stuck at 100%**: Check `TRAIN_TASKS` dictionary via `/train_status/<task_id>`; server logs may show save errors. ZIP may still have model files in `models/`.
- **Profile image not updating**: Ensure uploaded file saved to `static/profile/` and DB `profile_photo` stores the relative path. Templates should use `url_for('static', filename=profile_photo)` or pass `profile_photo` already resolved.

---

## Security & Production Notes

- **Passwords** are hashed with Werkzeug; still use HTTPS in production.
- **Secret key**: change `app.secret_key` via environment variable `FLASK_SECRET`.
- **Database**: SQLite is suitable for testing; use PostgreSQL/MySQL in production.
- **Spark**: Running Spark locally uses substantial memory — tune Spark settings and limits in production or use a managed Spark cluster.

---

## Extending & Customizing

- Add more Chart.js plugins by including CDN scripts in `visualize.html`.
- Add new ML algorithms by importing the estimator from `pyspark.ml` and adding to the model selection code.
- Replace background thread training with Celery/RQ + Redis for more robust job handling.

---

## License

This project is provided as-is for educational and prototyping purposes.

---

## Contact / Contribution

This project is made by Dhanush Gowda S 