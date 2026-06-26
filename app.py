from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import bcrypt

app = Flask(__name__)
app.secret_key = "your_secret_key"


# -----------------------------------------------------
# 1) Convert DB tuple → dictionary (MUST BE FIRST)
# -----------------------------------------------------
def user_to_dict(row):
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "password_hash": row[3],
        "role": row[4],
        "phone": row[5],
        "department": row[6],
        "created_at": row[7]
    }


# -----------------------------------------------------
# 2) Init DB (MUST be BEFORE login)
# -----------------------------------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'User',
            phone TEXT DEFAULT '',
            department TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# RUN INIT ONLY ONCE AT START
init_db()


# -----------------------------------------------------
# 3) LOGIN
# -----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"].encode("utf-8")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password, user[3]):
            session["user"] = user_to_dict(user)
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


# -----------------------------------------------------
# 4) SIGNUP
# -----------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"].encode("utf-8")

        password_hash = bcrypt.hashpw(password, bcrypt.gensalt())

        try:
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users(name, email, password_hash)
                VALUES (?, ?, ?)
            """, (name, email, password_hash))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already exists")

    return render_template("signup.html")


# -----------------------------------------------------
# 5) DASHBOARD
# -----------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


# -----------------------------------------------------
# 6) PROFILE
# -----------------------------------------------------
@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("profile.html", user=session["user"])


# -----------------------------------------------------
# 7) EDIT PROFILE
# -----------------------------------------------------
@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]

    if request.method == "POST":
        new_name = request.form["name"]
        new_email = request.form["email"]
        new_role = request.form["role"]
        new_phone = request.form["phone"]
        new_department = request.form["department"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("""
            UPDATE users 
            SET name=?, email=?, role=?, phone=?, department=? 
            WHERE id=?
        """, (new_name, new_email, new_role, new_phone, new_department, user["id"]))
        conn.commit()

        cur.execute("SELECT * FROM users WHERE id=?", (user["id"],))
        updated_user = cur.fetchone()
        conn.close()

        session["user"] = user_to_dict(updated_user)
        return redirect(url_for("profile"))

    return render_template("edit_profile.html", user=user)


# -----------------------------------------------------
# 8) LOGOUT
# -----------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------------------------------
# 9) RUN FLASK
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
