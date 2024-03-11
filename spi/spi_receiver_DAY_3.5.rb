# Welcome to Sonic Pi

# Sonic Pi OSC Receiver

# Define the port number to listen for OSC messages
set :osc_port, 4560

# Function to receive OSC messages and trigger synth behaviors
live_loop :osc_receiver do
  use_real_time
  address, *message = sync "/osc*/gesture"
  
  # Extract the gesture data
  gesture = message
  print gesture
  
  # React to different gestures
  case gesture
  when "Open Hand Detected"
    print
    # Synth behavior for open hand
    with_fx :reverb, room: 0.7 do
      play :C4, release: 0.5
      ##| sleep 0.5
    end
  when "Victory Sign Detected"
    # Synth behavior for victory sign
    sample :loop_amen
    sleep sample_duration :loop_amen
    
  when "Thumbs Up Detected"
    # Synth behavior for thumbs up
    use_synth :tri
    play :G3, release: 0.3
    ##| sleep 0.3
    # Add more cases for other gestures as needed
  end
end
