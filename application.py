from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# -----------------------------
# Load Ridge model and Scaler
# -----------------------------
with open("models/ridge.pkl", "rb") as f:
    ridge_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    standard_scaler = pickle.load(f)


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":

        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Scale input using loaded scaler
        new_data_scaled = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        # Predict
        result = ridge_model.predict(new_data_scaled)[0]

        return render_template("home.html", result=result)

    return render_template("home.html")


# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

