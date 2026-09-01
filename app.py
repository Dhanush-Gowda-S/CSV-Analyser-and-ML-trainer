import os
import uuid
import threading
import sqlite3
from pathlib import Path
from io import BytesIO
import zipfile
import json
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify, abort
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    current_user, UserMixin
)

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import StringType, DoubleType

# ML libs from Spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, Imputer
)
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODELS_FOLDER = BASE_DIR / "models"
DB_PATH = BASE_DIR / "app.db"
DEFAULT_AVATAR = "images/default_avatar.png"  # under static/

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(BASE_DIR / "static" / "profile", exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-in-prod")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

# Flask-Login
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# In-memory task tracking (task_id -> status dict)
TRAIN_TASKS = {}

# -------------------------
# Database init
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # users table includes is_admin and profile_photo
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        is_admin INTEGER DEFAULT 0,
        profile_photo TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        original_name TEXT,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        model_name TEXT,
        path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# safe shared connection for simple usage
def get_conn():
    return sqlite3.connect(DB_PATH)

# -------------------------
# User class + loader
# -------------------------
class User(UserMixin):
    def __init__(self, id_, username, email, is_admin=False, profile_photo=None):
        self.id = id_
        self.username = username
        self.email = email
        self.is_admin = bool(is_admin)
        self.profile_photo = profile_photo
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, username, email, is_admin, profile_photo FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2], row[3], row[4])
    return None

# -------------------------
# Misc helpers
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def log_action(user_id, action):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO logs (user_id, action) VALUES (?, ?)", (user_id, action))
    conn.commit()
    conn.close()

def get_profile_photo(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT profile_photo FROM users WHERE id=?", (user_id,))
    r = c.fetchone()
    conn.close()
    if r and r[0]:
        return r[0]
    return DEFAULT_AVATAR

# -------------------------
# SPARK session
# -------------------------
SPARK = (
    SparkSession.builder
    .master("local[*]")
    .appName("SparkML_Flask_App")
    .config("spark.sql.repl.eagerEval.enabled", "false")
    .getOrCreate()
)
SPARK.sparkContext.setLogLevel("ERROR")
# =========================
# app.py  (PART 2 of 4)
# =========================

# ---------- index / register / login / logout ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            flash("All fields required")
            return render_template("register.html")
        hashed = generate_password_hash(password)
        conn = get_conn()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
            conn.commit()
            flash("Registered — please login")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT id, username, email, password, is_admin, profile_photo FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[3], password):
            user = User(row[0], row[1], row[2], row[4], row[5])
            login_user(user)
            log_action(user.id, "login")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    log_action(current_user.id, "logout")
    logout_user()
    return redirect(url_for("index"))

# ---------- dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, filename, original_name, uploaded_at FROM files WHERE user_id=? ORDER BY uploaded_at DESC", (current_user.id,))
    files = c.fetchall()
    c.execute("SELECT id, model_name, path, created_at FROM models WHERE user_id=? ORDER BY created_at DESC", (current_user.id,))
    models = c.fetchall()
    c.execute("SELECT action, ts FROM logs WHERE user_id=? ORDER BY ts DESC LIMIT 30", (current_user.id,))
    logs = c.fetchall()
    conn.close()
    photo = get_profile_photo(current_user.id)
    return render_template("dashboard.html", files=files, models=models, logs=logs, profile_photo=photo)

# ---------- analyze (upload) ----------
@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        f = request.files.get("csvfile")
        if not f or f.filename == "":
            flash("Upload a CSV file")
            return redirect(request.url)
        if not allowed_file(f.filename):
            flash("Only CSV files are allowed")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        save_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{filename}"
        f.save(str(save_path))
        conn = get_conn()
        c = conn.cursor()
        c.execute("INSERT INTO files (user_id, filename, original_name) VALUES (?, ?, ?)", (current_user.id, str(save_path), filename))
        file_id = c.lastrowid
        conn.commit()
        conn.close()
        log_action(current_user.id, f"uploaded:{filename}")
        return redirect(url_for("analyze_result", file_id=file_id))
    return render_template("analyze.html")

MODEL_GUIDELINES = {
    "regression": "Regression models predict numeric values. LR = baseline, RF = robust, GBT = high accuracy.",
    "classification": "Classification models predict categories. Logistic = baseline, RF = robust, GBT = powerful."
}

# ---------- analyze_result (EDA)
@app.route("/analyze_result/<int:file_id>")
@login_required
def analyze_result(file_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT filename, original_name FROM files WHERE id=? AND user_id=?", (file_id, current_user.id))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("File not found")
        return redirect(url_for("dashboard"))
    filepath, original_name = row

    # load with spark
    try:
        df = SPARK.read.option("header", True).option("inferSchema", True).csv(filepath)
    except Exception as e:
        flash(f"Error reading CSV: {e}")
        return redirect(url_for("analyze"))

    # sanitize colnames
    new_cols = []
    for c_old in df.columns:
        c_new = c_old.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")
        new_cols.append(c_new)
    df = df.toDF(*new_cols)

    # drop empty or constant columns
    drop_cols = []
    for name in df.columns:
        non_null = df.filter(df[name].isNotNull()).count()
        if non_null == 0:
            drop_cols.append(name); continue
        distinct = df.select(name).distinct().count()
        if distinct <= 1:
            drop_cols.append(name)
    if drop_cols:
        df = df.drop(*drop_cols)

    total_rows = df.count()

    # detect numeric vs string
    numeric_cols = []
    string_cols = []
    for f in df.schema:
        t = f.dataType.simpleString()
        if t in ("double","float","int","long","short"):
            numeric_cols.append(f.name)
        else:
            string_cols.append(f.name)

    missing = {col: int(df.filter(df[col].isNull()).count()) for col in df.columns}

    # suggestion for ML targets - numeric excluding index-like names
    ml_targets = [c for c in numeric_cols if not any(k in c.lower() for k in ("index","id","sno"))]

    can_train = (len(ml_targets) > 0) and (total_rows >= 5)

    # basic correlations (limit to first 12)
    corrs = {}
    nc = numeric_cols[:12]
    for i in range(len(nc)):
        for j in range(i+1, len(nc)):
            a = nc[i]; b = nc[j]
            try:
                corrs[f"{a}___{b}"] = float(df.stat.corr(a, b) or 0.0)
            except:
                corrs[f"{a}___{b}"] = None

    # sample JSON for visualize (first 2000 rows)
    import pandas as pd
    sample_json = []
    try:
        sample_df = pd.read_csv(filepath, nrows=2000)
        sample_df.columns = [c.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "") for c in sample_df.columns]
        sample_json = sample_df.fillna("").to_dict(orient="records")
    except Exception:
        sample_json = []

    photo = get_profile_photo(current_user.id)

    return render_template(
        "analyze_result.html",
        file_id=file_id,
        filename=original_name,
        cols=df.columns,
        schema=[(f.name, f.dataType.simpleString()) for f in df.schema],
        corrs=corrs,
        missing=missing,
        numeric_cols=numeric_cols,
        string_cols=string_cols,
        ml_targets=ml_targets,
        can_train=can_train,
        model_guidelines=MODEL_GUIDELINES,
        profile_photo=photo,
        sample_data=sample_json
    )

# ---------- visualize endpoint (client-side uses sample_data) ----------
@app.route("/visualize/<int:file_id>")
@login_required
def visualize(file_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT filename, original_name FROM files WHERE id=? AND user_id=?", (file_id, current_user.id))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("File not found")
        return redirect(url_for("dashboard"))
    filepath, original_name = row

    import pandas as pd
    try:
        df = pd.read_csv(filepath, nrows=5000)
        df.columns = [c.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "") for c in df.columns]
        data = df.fillna("").to_dict(orient="records")
    except Exception as e:
        flash(f"Error preparing visualization: {e}")
        data = []

    photo = get_profile_photo(current_user.id)
    return render_template("visualize.html", filename=original_name, cols=list(df.columns) if len(data)>0 else [], data=data, profile_photo=photo)
# =========================
# app.py  (PART 3 of 4)
# =========================

@app.route("/start_train", methods=["POST"])
@login_required
def start_train():
    # Accept form-data
    data = request.form if request.form else request.get_json() or {}
    file_id = data.get("file_id") or request.form.get("file_id")
    target = data.get("target") or request.form.get("target")
    model_mode = data.get("model_mode") or request.form.get("model_mode") or "B"
    model_type = data.get("model_type") or request.form.get("model_type") or None
    model_choice = data.get("model_choice") or request.form.get("model_choice") or None
    model_name = data.get("model_name") or request.form.get("model_name") or f"model_{uuid.uuid4().hex[:6]}"
    include_categorical = str(data.get("include_categorical") or request.form.get("include_categorical") or "true").lower() in ("1","true","yes","on")
    selected_features = data.get("selected_features") or request.form.get("selected_features") or ""

    try:
        file_id = int(file_id)
    except:
        return jsonify({"error":"invalid file_id"}), 400

    task_id = uuid.uuid4().hex
    TRAIN_TASKS[task_id] = {"status":"queued","progress":0,"result":None}
    t = threading.Thread(target=_train_job, args=(task_id, current_user.id, file_id, target, model_mode, model_type, model_choice, model_name, include_categorical, selected_features), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

def _train_job(task_id, user_id, file_id, target_col, model_mode, model_type, model_choice, model_name, include_categorical, selected_features):
    """
    Universal training job with safe fallbacks.
    Updates TRAIN_TASKS[task_id] with progress/status/result.
    The final result will include:
      - status: "done" or "failed"
      - progress (0..100)
      - result.metrics (dict)
      - result.feature_importance (list of {feature, value})
      - We also set keys feature_importances and feature_importance for compatibility
    """
    try:
        TRAIN_TASKS[task_id].update({"status":"running","progress":5})
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT filename FROM files WHERE id=?", (file_id,))
        r = c.fetchone()
        conn.close()
        if not r:
            TRAIN_TASKS[task_id].update({"status":"failed","progress":0,"result":{"error":"file not found"}})
            return
        csv_path = r[0]
        TRAIN_TASKS[task_id]["progress"] = 10

        # read dataset
        sdf = SPARK.read.option("header", True).option("inferSchema", True).csv(csv_path)
        # clean names
        for old in sdf.columns:
            new = old.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")
            if new != old:
                sdf = sdf.withColumnRenamed(old, new)

        # drop all-null/constant cols
        drop_cols = []
        for name in sdf.columns:
            non_null = sdf.filter(sdf[name].isNotNull()).count()
            if non_null == 0:
                drop_cols.append(name); continue
            distinct = sdf.select(name).distinct().count()
            if distinct <= 1:
                drop_cols.append(name)
        if drop_cols:
            sdf = sdf.drop(*drop_cols)

        TRAIN_TASKS[task_id]["progress"] = 20

        # dynamic type detection
        numeric_cols = []
        string_cols = []
        for f in sdf.schema:
            t = f.dataType.simpleString()
            if t in ("double","float","int","long","short"):
                numeric_cols.append(f.name)
            else:
                string_cols.append(f.name)

        # guard target
        if not target_col or target_col not in sdf.columns:
            TRAIN_TASKS[task_id].update({"status":"failed","progress":0,"result":{"error":f"target '{target_col}' not found"}})
            return

        # cast numeric columns to double
        for nc in numeric_cols:
            try:
                sdf = sdf.withColumn(nc, regexp_replace(col(nc).cast("string"), r"^\s*$", None))
                sdf = sdf.withColumn(nc, col(nc).cast("double"))
            except:
                pass

        # prepare label column
        label_col = "label"
        # determine target dtype
        tgt_dtype = [f.dataType.simpleString() for f in sdf.schema if f.name == target_col][0]
        if tgt_dtype.startswith("string"):
            si = StringIndexer(inputCol=target_col, outputCol=label_col, handleInvalid="skip")
            sdf = si.fit(sdf).transform(sdf)
            is_classification = True
        else:
            sdf = sdf.withColumn(target_col, col(target_col).cast("double"))
            sdf = sdf.withColumnRenamed(target_col, label_col)
            is_classification = False

        TRAIN_TASKS[task_id]["progress"] = 35

        sdf = sdf.dropna(subset=[label_col])

        # feature selection per mode
        numeric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in ("index","id","sno")) and c != target_col]
        chosen_numeric = []
        chosen_string = []

        if model_mode == "A":
            chosen_numeric = numeric_cols
            if include_categorical:
                chosen_string = string_cols
        elif model_mode == "B":
            # correlation-based selection
            chosen_numeric = []
            for colname in numeric_cols:
                try:
                    corr = sdf.stat.corr(colname, label_col)
                    if corr is None: continue
                    if abs(float(corr)) >= 0.20:
                        chosen_numeric.append(colname)
                except:
                    continue
            if not chosen_numeric:
                chosen_numeric = numeric_cols
            if include_categorical:
                chosen_string = string_cols
        else:
            req = [s.strip() for s in (selected_features or "").split(",") if s.strip()]
            for r in req:
                if r in numeric_cols: chosen_numeric.append(r)
                elif r in string_cols: chosen_string.append(r)
            if not chosen_numeric:
                chosen_numeric = numeric_cols
            if include_categorical and not chosen_string:
                chosen_string = string_cols

        if not chosen_numeric and not chosen_string:
            TRAIN_TASKS[task_id].update({"status":"failed","progress":0,"result":{"error":"no features available"}})
            return

        TRAIN_TASKS[task_id]["progress"] = 50

        # impute numeric
        if chosen_numeric:
            imp = Imputer(inputCols=chosen_numeric, outputCols=[c + "_imp" for c in chosen_numeric]).setStrategy("mean")
            sdf = imp.fit(sdf).transform(sdf)
            chosen_numeric = [c + "_imp" for c in chosen_numeric]

        # indexers + encoders
        indexers = []
        encoders = []
        encoded_cols = []
        for s in chosen_string:
            si = StringIndexer(inputCol=s, outputCol=f"{s}_idx", handleInvalid="keep")
            ohe = OneHotEncoder(inputCol=f"{s}_idx", outputCol=f"{s}_vec")
            indexers.append(si); encoders.append(ohe); encoded_cols.append(f"{s}_vec")

        feature_cols = chosen_numeric + encoded_cols
        if not feature_cols:
            TRAIN_TASKS[task_id].update({"status":"failed","progress":0,"result":{"error":"no feature columns after encoding"}})
            return

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

        TRAIN_TASKS[task_id]["progress"] = 65

        # choose model
        if model_type not in ("regression","classification"):
            model_type = "classification" if is_classification else "regression"

        if model_type == "regression":
            if model_choice == "LinearRegression":
                model = LinearRegression(featuresCol="features", labelCol=label_col)
                chosen_name = "LinearRegression"
            elif model_choice == "RandomForestRegressor":
                model = RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=100)
                chosen_name = "RandomForestRegressor"
            else:
                model = GBTRegressor(featuresCol="features", labelCol=label_col, maxIter=50)
                chosen_name = "GBTRegressor"
            is_class = False
        else:
            if model_choice == "LogisticRegression":
                model = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=50)
                chosen_name = "LogisticRegression"
            elif model_choice == "RandomForestClassifier":
                model = RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=100)
                chosen_name = "RandomForestClassifier"
            else:
                model = RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=100)
                chosen_name = "RandomForestClassifier"
            is_class = True

        TRAIN_TASKS[task_id]["progress"] = 75

        stages = indexers + encoders + [assembler, model]
        pipeline = Pipeline(stages=stages)

        train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)
        TRAIN_TASKS[task_id]["progress"] = 85

        fitted = pipeline.fit(train_df)
        TRAIN_TASKS[task_id]["progress"] = 92

        preds = fitted.transform(test_df)
        TRAIN_TASKS[task_id]["progress"] = 96

        metrics = {}
        if not is_class:
            try:
                rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(preds)
                r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(preds)
                metrics["rmse"] = float(rmse); metrics["r2"] = float(r2)
            except Exception as e:
                metrics["error_eval"] = str(e)
        else:
            try:
                acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy").evaluate(preds)
                prec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision").evaluate(preds)
                rec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall").evaluate(preds)
                metrics["accuracy"] = float(acc); metrics["precision"] = float(prec); metrics["recall"] = float(rec)
                try:
                    if preds.select(label_col).distinct().count() == 2:
                        auc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)
                        metrics["auc"] = float(auc)
                except:
                    metrics["auc"] = None
            except Exception as e:
                metrics["error_eval"] = str(e)

        TRAIN_TASKS[task_id]["progress"] = 98

        # feature importance best-effort
        feature_importances = []
        try:
            last = fitted.stages[-1]
            if hasattr(last, "featureImportances"):
                importances = list(last.featureImportances)
                mapping_names = feature_cols
                if len(importances) == len(mapping_names):
                    feature_importances = sorted([(mapping_names[i], float(importances[i])) for i in range(len(importances))], key=lambda x: x[1], reverse=True)
                else:
                    feature_importances = [(mapping_names[i] if i < len(mapping_names) else f"f{i}", float(importances[i])) for i in range(min(len(importances), len(mapping_names)))]
            elif hasattr(last, "coefficients"):
                coeffs = list(last.coefficients)
                feature_importances = [(feature_cols[i], float(coeffs[i])) for i in range(min(len(coeffs), len(feature_cols)))]
        except Exception:
            feature_importances = []

        # prepare result format
        feat_list = [{"feature": f, "value": v} for f, v in feature_importances]

        # save model
        model_dir = MODELS_FOLDER / f"{user_id}_{model_name}_{uuid.uuid4().hex[:6]}"
        try:
            fitted.save(str(model_dir))
            conn = get_conn()
            c = conn.cursor()
            c.execute("INSERT INTO models (user_id, model_name, path) VALUES (?, ?, ?)", (user_id, model_name, str(model_dir)))
            conn.commit()
            conn.close()
        except Exception as e:
            # still mark done but with save error
            TRAIN_TASKS[task_id].update({
                "status":"done",
                "progress":100,
                "result":{
                    "metrics": metrics,
                    "feature_importance": feat_list,
                    "feature_importances": feat_list,
                    "save_error": str(e)
                }
            })
            log_action(user_id, f"trained:{model_name}")
            return

        # final success
        TRAIN_TASKS[task_id].update({
            "status":"done",
            "progress":100,
            "result":{
                "metrics": metrics,
                "feature_importance": feat_list,
                "feature_importances": feat_list,
                "model_name": model_name,
                "chosen_model": chosen_name
            }
        })
        log_action(user_id, f"trained:{model_name}")
    except Exception as e:
        TRAIN_TASKS[task_id].update({"status":"failed","progress":0,"result":{"error":str(e)}})
# =========================
# app.py  (PART 4 of 4)
# =========================

@app.route("/train_status/<task_id>")
@login_required
def train_status(task_id):
    task = TRAIN_TASKS.get(task_id)
    if not task:
        return jsonify({"status":"unknown","progress":0})
    # return both top-level fields and nested result for convenience
    response = {
        "status": task.get("status"),
        "progress": task.get("progress", 0)
    }
    # if detailed result exists expose flattened keys expected by JS
    if task.get("result"):
        response.update(task["result"])
    return jsonify(response)

# Download model (zip spark model dir)
@app.route("/download_model/<int:model_id>")
@login_required
def download_model(model_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT model_name, path, user_id FROM models WHERE id=?", (model_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("Model not found"); return redirect(url_for("dashboard"))
    model_name, path, owner = row
    if current_user.id != owner and not current_user.is_admin:
        flash("Unauthorized"); return redirect(url_for("dashboard"))
    mem = BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, path)
                zf.write(full, arc)
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"{model_name}.zip")

# profile route
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        f = request.files.get("profile_photo")
        photo_name = None
        if f and f.filename:
            fname = secure_filename(f.filename)
            photo_name = f"profile/{current_user.id}_{uuid.uuid4().hex}_{fname}"
            save_path = BASE_DIR / "static" / photo_name
            f.save(str(save_path))
        conn = get_conn()
        c = conn.cursor()
        if photo_name:
            c.execute("UPDATE users SET username=?, email=?, profile_photo=? WHERE id=?", (username, email, photo_name, current_user.id))
        else:
            c.execute("UPDATE users SET username=?, email=? WHERE id=?", (username, email, current_user.id))
        conn.commit()
        conn.close()
        log_action(current_user.id, "profile_updated")
        flash("Updated")
        # After updating DB
        updated = load_user(current_user.id)
        login_user(updated)  
        return redirect(url_for("profile"))
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT username, email, profile_photo FROM users WHERE id=?", (current_user.id,))
    row = c.fetchone()
    conn.close()
    photo = row[2] if row and row[2] else DEFAULT_AVATAR
    return render_template("profile.html", user=row, profile_photo=photo)

# Admin panel and admin management (create/promote/demote)
@app.route("/admin")
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash("Not authorized"); return redirect(url_for("dashboard"))
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, username, email, is_admin, profile_photo FROM users ORDER BY id ASC")
    users = c.fetchall()
    c.execute("SELECT user_id, action, ts FROM logs ORDER BY ts DESC LIMIT 200")
    logs = c.fetchall()
    conn.close()
    return render_template("admin.html", users=users, logs=logs)

# Create new admin user (form)
@app.route("/admin/create_admin", methods=["GET", "POST"])
@login_required
def admin_create_admin():
    if not current_user.is_admin:
        flash("Not authorized"); return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username","").strip()
        email = request.form.get("email","").strip()
        password = request.form.get("password","")
        if not username or not email or not password:
            flash("All fields required")
            return redirect(request.url)
        hashed = generate_password_hash(password)
        conn = get_conn()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, 1)", (username, email, hashed))
            conn.commit()
            flash("Admin created")
            log_action(current_user.id, f"created_admin:{username}")
            return redirect(url_for("admin_panel"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists")
        finally:
            conn.close()
    return render_template("admin_create.html")

# Promote user to admin
@app.route("/admin/promote/<int:uid>", methods=["POST"])
@login_required
def admin_promote(uid):
    if not current_user.is_admin:
        flash("Not authorized"); return redirect(url_for("dashboard"))
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE users SET is_admin=1 WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    log_action(current_user.id, f"promote:{uid}")
    flash("User promoted to admin")
    return redirect(url_for("admin_panel"))

# Demote admin to user
@app.route("/admin/demote/<int:uid>", methods=["POST"])
@login_required
def admin_demote(uid):
    if not current_user.is_admin:
        flash("Not authorized"); return redirect(url_for("dashboard"))
    # Prevent demoting yourself accidentally
    if uid == int(current_user.id):
        flash("You cannot demote yourself")
        return redirect(url_for("admin_panel"))
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE users SET is_admin=0 WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    log_action(current_user.id, f"demote:{uid}")
    flash("Admin demoted to user")
    return redirect(url_for("admin_panel"))

# Delete user (admin)
@app.route("/admin/delete_user/<int:uid>", methods=["POST"])
@login_required
def admin_delete_user(uid):
    if not current_user.is_admin:
        flash("Not authorized"); return redirect(url_for("dashboard"))
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (uid,))
    c.execute("DELETE FROM files WHERE user_id=?", (uid,))
    c.execute("DELETE FROM models WHERE user_id=?", (uid,))
    c.execute("DELETE FROM logs WHERE user_id=?", (uid,))
    conn.commit()
    conn.close()
    log_action(current_user.id, f"deleted_user:{uid}")
    flash("User deleted")
    return redirect(url_for("admin_panel"))

# Ensure an admin account exists and run
if __name__ == "__main__":
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE is_admin=1")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username,email,password,is_admin) VALUES (?, ?, ?, 1)",
                  ("admin", "admin@example.com", generate_password_hash("adminpass")))
        conn.commit()
        print("Default admin user created: admin / adminpass (change immediately)")
    conn.close()
    app.run(debug=True, port=5000)
