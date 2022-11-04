import smtplib


class Notification:
    def __init__(self):
        self.email_user = ''
        self.server = smtplib.SMTP ('', )
        self.server.starttls()
        self.server.login(self.email_user, '')

    def define_message(self, message, failed=False):
        if failed:
            self.message = f'taining failed for {message}'
        else:
            self.message = f'training done for {message}'

    def send_mail(self):
        self.server.sendmail(self.email_user, self.email_user, self.message)
        self.server.quit()
