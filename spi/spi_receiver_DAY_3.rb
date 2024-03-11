##| in_thread do
##|   loop do
##|     cue [:foo, :bar, :baz].choose
##|     sample :loop_amen
##|     sleep sample_duration :loop_amen
##|   end
##| end

##| live_loop :foo do


##|   use_real_time
##|   use_osc "10.229.65.137", 4560


##|   in_thread do
##|     loop do
##|       cue :tick
##|       sample :loop_amen
##|       sleep sample_duration :loop_amen
##|       ##| play :E
##|       ##| sleep 0.25
##|     end
##|   end

##|   loop do
##|     use_synth :bass_foundation
##|     sync :tick
##|     play :E-36, decay: 0.2, decay_level: 0.6, pan: 0, attack: 0, attack_level: 0, sustain: 0.1, sustain_level: 0.4 , release: 0.5
##|     sleep 1
##|     play :E-24, decay: 0.2, decay_level: 0.6, pan: 0, attack: 0, attack_level: 0, sustain: 0.1, sustain_level: 0.4 , release: 0.5
##|     sleep 0.25
##|     ##| use_synth :saw
##|     ##| play :E, decay: 0.1, pan: -1*i
##|     ##| sleep 0.25
##|     play :E-2
##|     sleep 0.25
##|     ##| sample :ambi_lunar_land
##|   end

##|   ##| loop do
##|   ##|   cue :tick
##|   ##|   sample :bd_haus, rate: 1
##|   ##|   sleep sample_duration :loop_amen
##|   ##| end
##| end

'''
THIS IS THE CALL AND RESPONSE PROTOCOL THAT PINGS 4 TIMES
SENDER OSC TO RECEIVER
RECEIVER OSC ACK RETURN
SENDER PRINT ACK TO CONSOLE
'''

##| define :ping do
##|   use_osc "10.229.64.136", 4560 # IP address and port of the receiver instance
##|   ##| Example OSC messages to control the tb303 synth
##|   ##| osc "/trigger/tb303", :e3, 70, 7, 1, 4, 4
##|   ##| osc "/trigger/prophet", :E1, 100, 0.25
##|   ##| osc "/trigger/baz"
##|   play 80
##|   sleep 0.15
##| end


##| in_thread(name: :c_r) do
##|   print "STARTING CALL AND RESPONSE"
##|   live_loop :live_1 do
##|     print "STARTING LOOP 1"
##|     cue :cue_1
##|     4.times do
##|       print "LOOP 1"
##|       ping
##|     end
##|     sleep 10
##|   end
##| end

##| in_thread(name: :c_r_2) do
##|   ##| THIS IS A LIVE LOOP WAITING FOR RESPONSE FROM
##|   live_loop :live_2 do
##|     print "LOOP 2"
##|     sync "/osc*/trigger/received"
##|     sync :cue_1

##|     with_fx :reverb do
##|       live_audio :lv_a, :stop
##|       play 80
##|       sample :ambi_piano
##|     end
##|   end
##| end

##| THIS WILL ONLY RUN ONCE BECAUSE ITS SYNCED TO EVERY TIME LIVE LOOP 1 RUNS
##| DIFFERENT FROM LIVE LOOP 2 IN THAT IT DOESNT TRACK EVERY MESSAGE ACKNOWLEDGEMENT, JUST THAT THE THREAD HAS BEEN STARTED
##| loop do
##|   print "EXTERNAL LOOP"
##|   sync :cue_1
##|   play 100
##| end


''
'
MIDI CONTROL OF LOCAL SONIC-PI SYNTH
'''

live_loop :local_instrument do
  
  use_real_time
  note_1, velocity_1 = sync "/midi:arturia_keystep_32:3/note_on"
  synth :fm, note: note_1, amp: velocity_1 / 127.0, decay: 0.25
  synth :saw, note: note_1 + 12, amp: velocity_1 / 127.0, decay: 0.25
  
end


'''
MIDI CONTROL OF THE JUNO
'''

live_loop :midi_piano do
  use_real_time
  
  note, velocity = sync "/midi:arturia_keystep_32:1/note_on"
  ##| synth :pluck, note: note, amp: velocity / 127, cutoff: 100, decay: 0.25
  
  use_osc "10.229.64.136", 4560
  osc "/trigger/juno", note, velocity
end

live_loop :midi_piano_2 do
  use_real_time
  
  note, velocity = sync "/midi:arturia_keystep_32:2/note_on"
  ##| synth :pluck, note: note, amp: velocity / 127, cutoff: 100, decay: 0.25
  
  use_osc "10.229.64.136", 4560
  osc "/trigger/md2", note, velocity
end

live_loop :start_loop do
  use_osc "10.229.64.136", 4560
  button, property = sync "/midi:arturia_keystep_32:1/control_change"
  
  
end


'''
INTERFACE WITH PYTHON FLASK SERVER
'''

live_loop :prophet_trigger do
  use_real_time
  # Wait for incoming OSC messages
  address, *args = sync "/osc*/trigger/prophet"
  
  # Process the incoming OSC messages
  if address == "/trigger/prophet"
    # Extract arguments (assuming note and velocity)
    note, velocity = args[0], args[1]
    
    
    # Play the prophet synth
    synth :prophet, note: note, amp: velocity / 127.0
  end
end



