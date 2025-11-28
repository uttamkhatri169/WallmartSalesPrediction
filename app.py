from flask import Flask, render_template, request, redirect, session
import pandas as pd
import joblib
import os
import plotly.graph_objs as go

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Load model and scaler once
model = joblib.load("randomforest1_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/manageinventory')
def manage_inventory():
    return render_template("manageinventory.html")

@app.route('/predictinventory', methods=['GET', 'POST'])
def predict_inventory():
    prediction = None
    show_chart = False
    chart_url = None

    if request.method == 'POST':
        try:
            # Extract form inputs
            store = int(request.form['Store'])
            dept = int(request.form['Dept'])
            is_holiday = int(request.form['IsHoliday'])
            size = float(request.form['Size'])
            year = int(request.form['Year'])
            quarter = int(request.form['Quarter'])
            week_of_year = int(request.form['WeekOfYear'])

            # Validate input
            if store < 0 or dept < 0 or size < 0:
                raise ValueError("Store, Dept, and Size must be non-negative.")

            # Prepare input DataFrame
            input_df = pd.DataFrame([[store, dept, is_holiday, size, year, quarter, week_of_year]],
                                    columns=["Store", "Dept", "IsHoliday", "Size", "Year", "Quarter", "WeekOfYear"])
            input_df_scaled = scaler.transform(input_df)
            prediction = round(model.predict(input_df_scaled)[0], 2)

            # Store values in session
            session['predicted_sales'] = prediction
            session['store'] = store
            session['dept'] = dept

            # Create Plotly chart
            fig = go.Figure(go.Bar(
                x=["Predicted Inventory"],
                y=[prediction],
                marker_color='orange'
            ))
            fig.update_layout(title="Predicted Inventory", yaxis_title="Sales")

            os.makedirs("static", exist_ok=True)
            chart_path = "static/prediction_plot.html"
            fig.write_html(chart_path)

            show_chart = True
            chart_url = chart_path

        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = f"⚠️ Error during prediction: {str(e)}"

    return render_template("predictinventory.html", prediction=prediction, show_chart=show_chart, chart_url=chart_url)

@app.route('/inventorymanagement', methods=['GET', 'POST'])
def inventory_management():
    inventory_result = None
    predicted_sales = session.get('predicted_sales', None)
    store = session.get('store', None)
    dept = session.get('dept', None)

    if request.method == 'POST':
        try:
            if predicted_sales is None or store is None or dept is None:
                raise ValueError("Missing prediction data. Please predict first.")

            current_inventory = int(request.form['current_inventory'])

            if current_inventory < 0:
                raise ValueError("Current Inventory must be non-negative.")

            predicted_sales = int(float(predicted_sales))
            inventory_needed = predicted_sales - current_inventory

            if inventory_needed > 20:
                status = "Reorder"
                status_class = "status-reorder"
            elif inventory_needed < -20:
                status = "Overstock"
                status_class = "status-overstock"
            else:
                status = "OK"
                status_class = "status-ok"

            inventory_result = {
                "store": store,
                "dept": dept,
                "predicted_sales": predicted_sales,
                "current_inventory": current_inventory,
                "inventory_needed": inventory_needed,
                "status": status,
                "status_class": status_class
            }

        except Exception as e:
            print(f"Inventory error: {e}")
            inventory_result = {
                "error": f"⚠️ Error: {str(e)}"
            }

    return render_template("inventorymanagement.html", inventory_result=inventory_result,
                           predicted_sales=predicted_sales, store=store, dept=dept)

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/reset', methods=['POST'])
def reset():
    session.pop('predicted_sales', None)
    session.pop('store', None)
    session.pop('dept', None)
    return redirect('/predictinventory')

if __name__ == '__main__':
    app.run(debug=True)
