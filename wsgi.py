from app import app as application

if __name__=='__main__':
	application.run('0.0.0.0',port='5005',debug=True)