from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("carprice_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    present_price = float(request.form["present_price"])
    kms_driven = int(request.form["kms_driven"])
    owner = int(request.form["owner"])

    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]

    # Encode categorical inputs
    fuel = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
    seller = {"Dealer": 0, "Individual": 1}[seller_type]
    trans = {"Manual": 0, "Automatic": 1}[transmission]

    # Final input in SAME ORDER as training
    input_data = np.array([[year, present_price, kms_driven,
                            fuel, seller, trans, owner]])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Car Price: ₹ {round(prediction, 2)} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)
