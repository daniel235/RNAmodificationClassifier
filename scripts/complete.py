import smtplib, time

def send_email(message=None):
    # Send Email when script is complete
    SERVER = "sysbio02.informatics.iupui.edu"
    FROM = "Your server <mail@yourcompany.com>"
    TO = "daniel.acevedo01@utrgv.edu"
    SUBJECT = "The Script Has Completed"
    MSG = message

    # Prepare actual message
    MESSAGE = """\
    From: %s
    To: %s
    Subject: %s

    %s
    """ % (FROM, ", ".join(TO), SUBJECT, MSG)

    # Send the mail
    server = smtplib.SMTP(SERVER)
    server.sendmail(FROM, TO, MESSAGE)
    server.quit()