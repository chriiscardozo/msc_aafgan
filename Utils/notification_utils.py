from twilio.rest import Client
import subprocess

account_sid = "YOUR_ACCOUNT_SID"
auth_token = "YOUR_AUTH_TOKEN"
client = Client(account_sid, auth_token)

def get_hostname():
    return str(subprocess.check_output('hostname', shell=True))

def send_message(txt):
    global client

    txt = '[' + get_hostname() + '] ' + txt

    message = client.messages.create(
        to="YOUR_PHONE_NUMBER", 
        from_="TWILIO_NUMBER", # information from twilio
        body=txt)
    message.sid