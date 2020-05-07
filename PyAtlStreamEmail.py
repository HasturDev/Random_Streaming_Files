import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv

sender_email = "pythonatlantatest@gmail.com"
reciever_email = "pythonatlantatest@gmail.com"
password = input("Type password and press enter:")

message = MIMEMultipart("alternative")
message["Subject"] = "some testing nonsense"
message["From"] = sender_email
message["To"] = reciever_email

text = """\
    Whats up,
    are you doing well today
    are you staying safe
    if yes then good
    stay indoors 
    corona be murdering people
    """
html = """\
    <html>
        <body>
            <p>Hello my good compatriot<br>
            you good?
            yeah?
            ok
            <img src="https://avatars2.githubusercontent.com/u/1516577?s=280&v=4">
            a nice image from python Atlanta
             </p>
        </body>
    </html>
    """
part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

message.attach(part1)
message.attach(part2)

context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, reciever_email, message.as_string()
    )
