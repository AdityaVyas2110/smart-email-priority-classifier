from src.predict import predict_email

email = input("Enter email text: ")

result = predict_email(email)

print("Email Category:", result)
