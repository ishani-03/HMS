EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_USE_TLS = True
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_HOST_USER = 'medeasy.aorta@gmail.com'
EMAIL_HOST_PASSWORD = 'aorta123' #try to use a password without @ SMTP gives error
EMAIL_PORT = 587

# also check if you have allowed third party access to your email account