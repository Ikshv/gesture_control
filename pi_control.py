from pythonosc import udp_client

# Define the IP address and port of Sonic Pi
SONIC_PI_IP = '127.0.0.1'  # Local IP address
SONIC_PI_PORT = 4559  # Change this to the port on which Sonic Pi is listening for OSC messages

# Create an OSC client instance
osc_client = udp_client.SimpleUDPClient(SONIC_PI_IP, SONIC_PI_PORT)

# Function to send OSC messages to Sonic Pi
def send_osc_message(message):
    osc_client.send_message("/play", message)  # Adjust the OSC address ("/play") and message format as needed

# Now you can call the send_osc_message function to send messages to Sonic Pi
# For example:
send_osc_message("C4")
