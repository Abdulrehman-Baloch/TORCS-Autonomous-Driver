'''
Modified Driver class to use neural network model with improved gear and clutch handling
'''

import msgParser
import carState
import carControl
from ai_driver import TORCSAIDriver  # Import your AI driver

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage, model_path):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.model_path = model_path
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        self.prev_speed = None
        self.stuck_time = 0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.lap_start_time = 0
        self.current_lap = 0
        self.last_dist_from_start = 0
        self.best_lap_time = float('inf')
        self.debug_counter = 0
        
        # Initialize the AI driver
        try:
            self.ai_driver = TORCSAIDriver(self.model_path)
            self.using_ai = True
            print("Successfully loaded AI driver model")
        except Exception as e:
            self.using_ai = False
            print(f"Failed to load AI driver: {e}")
            print("Falling back to rule-based driving")
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        # Parse incoming message from server
        self.state.setFromMsg(msg)
        
        # Track lap times - safer implementation
        try:
            self.track_lap_times()
        except Exception as e:
            # Just log the error but continue
            if self.debug_counter % 100 == 0:  # Don't spam logs
                print(f"Warning in lap tracking: {e}")
        
        # Check for stuck condition
        self.check_stuck()
        
        # Increment debug counter for periodic outputs
        self.debug_counter += 1
        
        if self.using_ai and not self.recovery_mode:
            try:
                # Use AI to predict control commands
                control_dict = self.ai_driver.predict(self.state)
                
                # Set control values from AI
                self.control.setSteer(control_dict['steer'])
                self.control.setAccel(control_dict['accel'])
                self.control.setBrake(control_dict['brake'])
                
                # Handle gear based on AI prediction if available
                if 'gear' in control_dict:
                    self.control.setGear(int(control_dict['gear']))
                else:
                    self.gear()  # Fallback to rule-based gear selection
                
                # Handle clutch if available
                if 'clutch' in control_dict:
                    self.control.setClutch(control_dict['clutch'])
                
                # Debug output every 20 iterations
                if self.debug_counter % 20 == 0:
                    print(f"AI: Speed={abs(self.state.speedX):.1f}, RPM={self.state.rpm:.0f}, Gear={self.state.gear}, " +
                          f"Control: Steer={control_dict['steer']:.2f}, Accel={control_dict['accel']:.2f}, " +
                          f"Brake={control_dict['brake']:.2f}" +
                          (f", Gear={control_dict['gear']}" if 'gear' in control_dict else "") +
                          (f", Clutch={control_dict['clutch']:.2f}" if 'clutch' in control_dict else ""))
                
            except Exception as e:
                print(f"AI prediction failed: {e}")
                print("Falling back to rule-based driving")
                self.steer()
                self.gear()
                self.speed()
        elif self.recovery_mode:
            # Execute recovery maneuver if we're stuck
            self.execute_recovery()
        else:
            # Fallback to original rule-based driving
            self.steer()
            self.gear()
            self.speed()
        
        # Save current state info for next iteration
        self.prev_rpm = self.state.rpm
        self.prev_speed = self.state.speedX
        self.last_dist_from_start = self.state.distFromStart if hasattr(self.state, 'distFromStart') else 0
        
        # Send control commands back to server
        return self.control.toMsg()
    
    def track_lap_times(self):
        """Track and report lap times safely"""
        # Check if required attributes exist
        if not hasattr(self.state, 'lastLapTime') or not hasattr(self.state, 'distFromStart'):
            return
        
        # Detect lap completion by watching for track position reset
        # When crossing the finish line, distFromStart goes from a large value back to near zero
        current_dist = self.state.distFromStart
        
        # Detect lap completion - when we go from near the end of the track back to the start
        if self.last_dist_from_start > 0 and current_dist < self.last_dist_from_start - 1000:
            # We've likely crossed the finish line
            self.current_lap += 1
            lap_time = self.state.lastLapTime if hasattr(self.state, 'lastLapTime') else 0
            
            if lap_time > 0:  # Ignore zero/invalid lap times
                # Update best lap time
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    print(f"NEW BEST LAP: {lap_time:.3f}s")
                else:
                    print(f"Lap completed: {lap_time:.3f}s (Best: {self.best_lap_time:.3f}s)")
    
    def check_stuck(self):
        """Check if the car is stuck and needs recovery"""
        # Define stuck parameters
        speed_threshold = 3.0
        stuck_time_threshold = 100  # iterations
        
        current_speed = abs(self.state.speedX)
        
        # Check if car is moving very slowly
        if current_speed < speed_threshold:
            self.stuck_time += 1
        else:
            self.stuck_time = 0
            self.recovery_mode = False
        
        # If stuck for too long, enter recovery mode
        if self.stuck_time > stuck_time_threshold and not self.recovery_mode:
            print("Car is stuck! Entering recovery mode.")
            self.recovery_mode = True
            self.recovery_counter = 0
    
    def execute_recovery(self):
        """Execute a recovery maneuver to get unstuck"""
        # Simple recovery: reverse with steering opposite to current angle
        max_recovery_iterations = 100
        
        if self.recovery_counter < 30:
            # First back up
            self.control.setGear(-1)  # Reverse
            self.control.setSteer(-self.state.angle * 2)  # Opposite steering
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            self.control.setClutch(0.0)
        elif self.recovery_counter < 60:
            # Then go forward
            self.control.setGear(1)
            self.control.setSteer(self.state.angle * 2)  # Exaggerated steering
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            self.control.setClutch(0.0)
        else:
            # If still stuck after attempts, try more random movements
            if self.recovery_counter % 2 == 0:
                # Wiggle steering
                self.control.setSteer(-self.state.angle)
            else:
                self.control.setSteer(self.state.angle)
            
            self.control.setGear(1)
            self.control.setAccel(0.7)
            self.control.setBrake(0.0)
        
        self.recovery_counter += 1
        
        # Exit recovery mode after maximum attempts
        if self.recovery_counter > max_recovery_iterations:
            print("Exiting recovery mode.")
            self.recovery_mode = False
            self.stuck_time = 0
    
    def steer(self):
        # Enhanced rule-based steering
        angle = self.state.angle
        dist = self.state.trackPos
        
        # Basic steering formula
        steering = (angle - dist*0.5)/self.steer_lock
        
        # Apply steering corrections based on track sensors if available
        if self.state.track and len(self.state.track) >= 19:
            # Get track sensors
            left_sensor = self.state.track[0]  # 90 degrees left
            left_diag_sensor = self.state.track[3]  # 45 degrees left
            center_sensor = self.state.track[9]  # Center
            right_diag_sensor = self.state.track[15]  # 45 degrees right
            right_sensor = self.state.track[18]  # 90 degrees right
            
            # Adjust steering based on nearest obstacles
            speed = self.state.speedX
            if abs(speed) > 50:
                # At high speeds, look further ahead
                if left_diag_sensor < right_diag_sensor and left_diag_sensor < 50:
                    # Obstacle on left, steer right
                    steering -= 0.1
                elif right_diag_sensor < left_diag_sensor and right_diag_sensor < 50:
                    # Obstacle on right, steer left
                    steering += 0.1
        
        # Apply steering
        self.control.setSteer(max(-1.0, min(1.0, steering)))
    
    def gear(self):
        # Enhanced gear shifting logic with RPM curves
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = abs(self.state.getSpeedX())
        
        # Always start in 1st gear if stopped or in neutral
        if gear <= 0 and speed < 5:
            gear = 1
            self.control.setGear(gear)
            return
            
        # Remember previous RPM to determine if engine is accelerating or decelerating
        if self.prev_rpm is None:
            rpm_increasing = True
        else:
            rpm_increasing = (rpm > self.prev_rpm)
        
        # Dynamic gear shifting thresholds based on current conditions
        if rpm_increasing:
            # When accelerating, shift up at higher RPM for better performance
            upshift_rpm = 6800  # Lower threshold to ensure upshifts happen
            # Adjust upshift point based on acceleration
            if self.prev_speed is not None:
                acceleration = abs(speed - abs(self.prev_speed))
                if acceleration > 1.0:  # Strong acceleration
                    upshift_rpm = 7000  # Delay upshift for more power
                elif acceleration < 0.2:  # Weak acceleration
                    upshift_rpm = 6500  # Upshift earlier
        else:
            # When not accelerating, upshift at lower RPM for better economy
            upshift_rpm = 6500
        
        # Downshift threshold - keep engine in its power band
        downshift_rpm = 3500  # Higher threshold to prevent unnecessary downshifts
        
        # Shift up if RPM is too high
        if rpm > upshift_rpm and gear < 6:
            gear += 1
            print(f"Upshift to gear {gear} at RPM={rpm:.0f}")
        
        # Shift down if RPM is too low
        elif rpm < downshift_rpm and gear > 1:
            gear -= 1
            print(f"Downshift to gear {gear} at RPM={rpm:.0f}")
        
        # Special case for very low speeds
        if speed < 15 and gear > 1:
            gear = 1  # Stay in first gear at very low speeds
        
        # Print debug info every 40 iterations
        if self.debug_counter % 40 == 0:
            print(f"Rule-based gear: Speed={speed:.1f}, RPM={rpm:.0f}, Gear={gear}")
        
        # Set the new gear
        self.control.setGear(gear)
        
        # Apply clutch during gear changes
        if gear != self.state.gear:
            self.control.setClutch(0.5)  # Apply clutch during shifts
        else:
            self.control.setClutch(0.0)  # No clutch when staying in the same gear
    
    def speed(self):
        # Enhanced rule-based speed control with track awareness
        speed = abs(self.state.getSpeedX())
        
        # Default acceleration
        accel = self.control.getAccel()
        
        # Basic speed control based on maximum speed
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        # Apply clutch slip for starting from a stop
        if speed < 5 and self.state.gear == 1:
            # Manage clutch for smooth starts
            self.control.setClutch(max(0.5, 1.0 - (self.state.rpm / 3000)))
        
        # Adjust speed based on track sensors if available
        brake = 0.0
        
        if self.state.track and len(self.state.track) >= 19:
            # Calculate a safe speed based on what's ahead
            center_sensor = self.state.track[9]  # Center
            front_left_sensor = self.state.track[8]  # Slightly left
            front_right_sensor = self.state.track[10]  # Slightly right
            
            # Calculate minimum distance ahead - our safety margin
            min_distance = min(center_sensor, front_left_sensor, front_right_sensor)
            
            # If we're approaching a turn or obstacle, reduce speed
            safe_speed = 100.0  # Default safe speed
            
            if min_distance < 100:
                # Reduce safe speed based on distance to obstacle
                safe_speed = min_distance * 0.8  # Linear scaling
                
                # Apply brakes if we're going too fast for the conditions
                if speed > safe_speed and speed > 30:
                    brake_intensity = min(1.0, (speed - safe_speed) / 50.0)  # Proportional braking
                    brake = brake_intensity
                    accel = 0.0  # Don't accelerate while braking
        
        # Set acceleration and brake
        self.control.setAccel(accel)
        self.control.setBrake(brake)
            
    def onShutDown(self):
        # Report final stats
        if hasattr(self, 'best_lap_time') and self.best_lap_time < float('inf'):
            print(f"Session complete. Best lap time: {self.best_lap_time:.3f}s")
    
    def onRestart(self):
        self.prev_rpm = None
        self.prev_speed = None
        self.stuck_time = 0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.lap_start_time = 0
        self.current_lap = 0
        self.last_dist_from_start = 0
        self.best_lap_time = float('inf')
        self.debug_counter = 0