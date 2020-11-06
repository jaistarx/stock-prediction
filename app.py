import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


app = Flask(__name__)
from tensorflow.keras.models import load_model
import os
model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"stocks.h5"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/monts.html')
def goals():
    return render_template('monts.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    import random
    a = []
    for i in range(60, 80):
        num2 = random.randrange(int(float(int_features[0])) - 5, int(float(int_features[0])) + 10)
        a.append(num2)
    tss = pd.DataFrame([a])
    inputs = tss.values.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    inputs = sc.fit_transform(inputs)
    X_test = []
    X_test.append(inputs)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price[0])
    k = 0
    for i in range(5):
        l = random.randrange(predicted_stock_price[0] - 5, predicted_stock_price[0] + 2)
        k += l
    ans = k / 5

    return render_template('index.html', prediction_text= 'Stock Price: ${}'.format(ans))

@app.route('/monte', methods=['POST'])
def monte():
    avg = 1
    std_dev = .1
    num_reps = 500
    num_simulations = 1000
    pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)
    sales_target_values = [75_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    sales_target_prob = [.3, .3, .2, .1, .05, .05]
    sales_budget_available = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)

    int_features = [str(x) for x in request.form.values()]
    df = pd.DataFrame(index=range(num_reps), data={'Pct_To_Target': pct_to_target,
                                                   'Sales_Budget_Available': sales_budget_available})

    df['Actual_Sales'] = df['Pct_To_Target'] * df['Sales_Budget_Available']
    all_stats = []

    for i in range(num_simulations):
        Sales_Budget_Available = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)
        pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)

        df = pd.DataFrame(index=range(num_reps), data={'Pct_To_Target': pct_to_target,
                                                       'Sales_Budget_Available': Sales_Budget_Available})

        df['Actual_Sales'] = df['Pct_To_Target'] * df['Sales_Budget_Available']

        all_stats.append([df['Actual_Sales'].sum().round(0),
                          df['Sales_Budget_Available'].sum().round(0)])
        results_df = pd.DataFrame.from_records(all_stats, columns=['Actual_Sales',
                                                                   'Sales_Budget_Available'])

        results_df['sales_estimate'] = results_df['Actual_Sales'] - results_df['Sales_Budget_Available']
        s = 0
        for i in results_df['Sales_Budget_Available']:
            s = s + i

        p = s / 1000
        if int(int_features[0]) > p:
            answer='ok'
        else:
            answer='not okay'

        return render_template('monts.html', prediction_texts=answer)

if __name__ == "__main__":
    app.run(debug=True)