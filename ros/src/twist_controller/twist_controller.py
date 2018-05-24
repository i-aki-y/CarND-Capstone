import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import numpy as np

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.05
        kd = 0.1
        mn = 0.0 # Minimum throttle value
        mx = 0.2 # Maximu throttle value

        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        cte_kp = 0.6
        cte_ki = 0.01
        cte_kd = 0.5
        cte_mn = -1.5 # Minimum throttle value
        cte_mx = 1.5 # Maximu throttle value
        self.cte_pid = PID(cte_kp, cte_ki, cte_kd, cte_mn, cte_mx)

        tau = 0.5  # 1/(2pi*tau) = cutoff frequency
        ts = 0.02  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, closest_wp_xy, next_closest_wp_xy, current_pose_xy):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.cte_pid.reset()
            return 0., 0., 0.

        wp_vec = next_closest_wp_xy - closest_wp_xy
        wp_vec_rot = np.array([-wp_vec[1], wp_vec[0]]) # rotate 90 deg
        car_wp_vec = closest_wp_xy - current_pose_xy
        cte = np.dot(wp_vec_rot, car_wp_vec) / np.linalg.norm(wp_vec)

        current_vel = self.vel_lpf.filt(current_vel)

        # rospy.logwarn("Angular vel: {0}".format(angular_vel))
        # rospy.logwarn("Target velocity: {0}".format(linear_vel))
        # rospy.logwarn("Target angular velocity: {0}\n".format(angular_vel))
        # rospy.logwarn("Current velocity: {0}\n".format(current_vel))
        # rospy.logwarn("Filtered velocity: {0}\n".format(self.vel_lpf.get()))

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        cte_steering = self.cte_pid.step(cte, sample_time)
        yaw_steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        rospy.logwarn("yaw_cont:{0}, pid:{1}".format(yaw_steering, cte_steering))

        steering = yaw_steering + cte_steering

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400 #N*m - to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2
                        #(400 ~ vehicle_mass * wheel_radius)

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius  # Torque N*m

        return throttle, brake, steering
