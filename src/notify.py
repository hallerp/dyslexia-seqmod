import smtplib


class Notification:
    def __init__(self):
        self.email_user = 'patrick.hlr@gmail.com'
        self.server = smtplib.SMTP ('smtp.gmail.com', 587)
        self.server.starttls()
        self.server.login(self.email_user, 'wmlbdmujhphrycjv')

    def define_message(self, message, failed=False):
        if failed:
            self.message = f'taining failed for {message}'
        else:
            self.message = f'training done for {message}'

    def send_mail(self):
        self.server.sendmail(self.email_user, self.email_user, self.message)
        self.server.quit()
