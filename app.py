from flask import Flask, render_template, flash, request, session

import mysql.connector

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] ='789546321452145a'

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/adminlogin")
def adminlogin():
    return render_template('Adminlogin.html')

@app.route("/ADMINLOGIN", methods=['GET', 'POST'])
def ADMINLOGIN():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb ")
            data = cur.fetchall()
            flash("Your are Logged In...!")
            return render_template('AdminHome.html',data=data)
        else:
            flash("Username or Password is wrong")
            return render_template('Adminlogin.html')

@app.route("/userlogin")
def userlogin():
    return render_template('Userlogin.html')

@app.route("/USERLOGIN", methods=['GET', 'POST'])
def USERLOGIN():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['Password']
        session['username'] = request.form['username']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where UserName='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            flash("Username or Password is wrong...!")
            return render_template('UserLogin.html')
        else:
            session['mob'] = data[2]
            session['email'] = data[3]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where UserName='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()
            flash("Your are Logged In...!")
            return render_template('UserHome.html', data=data)




@app.route("/newuser")
def newuser():
    return render_template('NewUser.html')

@app.route("/NEWUSER", methods=['GET', 'POST'])
def NEWUSER():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']
        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where UserName='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            conn = mysql.connector.connect(user='root', password='', host='localhost',
                                           database='1cyberthreatdb')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO regtb VALUES ('','" + name + "','" + mobile + "','" + email + "','" + address + "','" + username + "','" + password + "')")
            conn.commit()
            conn.close()
            flash('New User register successfully')
            return render_template('Userlogin.html')
        else:
            flash('Already registered')
            return render_template('NewUser.html')



@app.route("/UserHome")
def UserHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where UserName='"+session['username']+"'")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)



@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1cyberthreatdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/Predict")
def Predict():
    return render_template('Predict.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        import tensorflow as tf
        import numpy as np
        import joblib

        # Load everything ONCE (better to move outside route in production)
        model = tf.keras.models.load_model("cnn_lstm_network_attack_model.h5")
        scaler = joblib.load("scaler.save")
        feature_encoders = joblib.load("label_encoders.save")
        attack_encoder = joblib.load("attack_type_encoder.save")

        # Input string from form
        input_data = request.form['name']
        values = input_data.split(',')

        # Encode categorical features
        values[0] = feature_encoders['Protocol'].transform([values[0]])[0]
        values[7] = feature_encoders['Flags'].transform([values[7]])[0]

        # Convert to float
        float_array = np.array(values, dtype=float).reshape(1, -1)

        # Scale
        scaled_input = scaler.transform(float_array)

        # ===============================
        # CREATE SEQUENCE (TIME_STEPS=15)
        # ===============================
        TIME_STEPS = 15

        # Repeat same row 15 times (since single prediction)
        sequence_input = np.repeat(
            scaled_input[np.newaxis, :, :],
            TIME_STEPS,
            axis=1
        )

        # Shape becomes (1, 15, features)

        # Predict
        pred_probs = model.predict(sequence_input)

        predicted_class_index = np.argmax(pred_probs, axis=1)[0]
        predicted_label = attack_encoder.inverse_transform([predicted_class_index])[0]

        confidence = float(np.max(pred_probs))
        import winsound
        filename = 'alert.wav'
        #winsound.PlaySound(filename, winsound.SND_FILENAME)
        result = f"Prediction: {predicted_label} (Confidence: {round(confidence*100,2)}%)"
        sendmsg(session['mob'],result)
        sendmail(session['email'],result)
        return render_template('Result.html', result=result)


def sendmsg(targetno,message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")



def sendmail(Mailid,message):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "studentmailsend@gmail.com"
    toaddr = Mailid

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = message

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "jfrj aazz krww zkoh")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
