import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database = "kelas_kendaraan"
)

mycursor = mydb.cursor()

def inserData(object_name, long_pixel,long_meter ):
    sql = """
    INSERT INTO result (object_name, long_pixel, long_meter) 
    VALUES (%s , %s, %s)
    """
    val = (object_name, long_pixel,long_meter)
    mycursor.execute(sql,val)
    mydb.commit()

def showData():
    sql = "SELECT * FROM result"

    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    for i in myresult:
        print(i)

# inserData("bicycle", 2000, 2)
showData()

