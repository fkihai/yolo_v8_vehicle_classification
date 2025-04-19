from onvif import ONVIFCamera

# Konfigurasi kamera ONVIF
IP_ADDRESS = "192.168.1.100"   # Ganti dengan IP kamera
PORT = 8000                    # Port ONVIF, biasanya 80
USERNAME = "admin"             # Username kamera
PASSWORD = "admin"          # Password kamera

# Hubungkan ke kamera
def connect_camera(ip, port, username, password):
    try:
        camera = ONVIFCamera(ip, port, username, password)
        print("Berhasil terhubung ke kamera ONVIF!")
        return camera
    except Exception as e:
        print(f"Gagal menghubungkan ke kamera: {e}")
        return None

camera = connect_camera(IP_ADDRESS, PORT, USERNAME, PASSWORD)

if camera:
    # Mendapatkan layanan media
    media_service = camera.create_media_service()

    # Mendapatkan profil media pertama
    profiles = media_service.GetProfiles()
    profile_token = profiles[0].token  # Gunakan token profil pertama

    # Siapkan konfigurasi StreamSetup
    stream_setup = {
        "Stream": "RTP-Unicast",  # Bisa juga "RTP-Multicast"
        "Transport": {"Protocol": "RTSP"}  # Menggunakan protokol RTSP
    }

    # Dapatkan URL stream
    try:
        stream_uri = media_service.GetStreamUri({
            "StreamSetup": stream_setup,
            "ProfileToken": profile_token
        })
        print(f"Stream URL: {stream_uri.Uri}")
    except Exception as e:
        print(f"Error mendapatkan Stream URI: {e}")
