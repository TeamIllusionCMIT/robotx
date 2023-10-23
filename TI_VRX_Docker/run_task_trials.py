# Task 1 - read the goal from /vrx/stationkeeping/goal - use differential drive
# style steering to reach the goal.

# Task 2 - maneuver through waypoints.

# Python3 brute force the waypoint selection as presented
# in a post to GeeksForGeeks
# https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
#
# Your goal is to do a better job than the naive approach.

# V is the number of waypoints, graph is the cost between all combinations
# of waypoints. In trial runs, VRX generated three waypoints, with positions
# shifting slightly each iteration, due to environmental conditions

# waypoint {x, y, z, w}
# 0 {-33.72276866,150.673990424,1.8}
# 1 {-33.72267666999, 150.6740630167, 0.5718675632060268, 0.8203459576013042}
# 2 {-33.7220832884, 150.6739127877, 0.479425538604203, 0.88775825618903728}
# 3 {-33.7226013209, 150.6767504609858, 0.479425538604203, 0.8775825618903728:

# The graph array for these measures would be the distance from each point to
# all of the others in a matrix:

# [ 0->0 0->1 0->2 0->3 ]
# [ 1->0 1->1 1->2 1->3 ]
# [ 2->0 2->1 2->2 2->3 ]
# [ 1->0 3->1 3->2 3->3 ]

# And then you want to determine the length of all of the possible paths

# 0->1 1->2 2->3
# 0->1 1->3 3->2
# 0->2 2->1 1->3
# 0->2 2->3 3->1
# 0->3 3->1 1->2
# 0->3 3->2 2->1

# Task 3 - identify the objects and publish their locations.
#
# You can read the indivudal frames from the wamv sensor defined for each
# camera - /wamv/sensors/cameras/middle_right_camera/image_raw - for instance
# for the middle right camera. Process the image and write the coordinates
# to a PoseStamped message in the /vrx/perception/landmark topic.

# Task 4 - steer toward an acoustic beacon

# subscribe to /wamv/pingers/pinger/range_bearing . This should work with the
# same code as stationkeeping, because you use the geographic coordinates
# in that task to compute a range and bearing and then drive to it.

# Task 5 - encounter wildlife and do not injure it

# You can subscribe to the location data and use that, but it is sent at
# a deliberately slow rate. You really want to use perception to find the
# animals. Alex has created code that goes to the animals in turn and is working
# on circling them.

# Task 6 -
#
#

# Task 7 -
#
#

# Task 8 -
#
#


import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import ParamVec
from std_msgs.msg import Float64
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image  # Image is the message type per Automaic Addison
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
from sys import maxsize
from itertools import permutations
import haversine as hs
from haversine import Unit
import math


class WallopingWindowBlind(Node):
    def __init__(self):
        print("Starting node...")
        super().__init__("walloping_window_blind")
        self.task_type = ""
        self.task_state = ""
        self.task_sub = self.create_subscription(
            ParamVec, "/vrx/task/info", self.taskCB, 10
        )

        self.wamv_latitude = 0.0
        self.wamv_longitude = 0.0
        self.wamv_altitude = 0.0
        self.wamv_sub = self.create_subscription(
            NavSatFix, "/wamv/sensors/gps/gps/fix", self.wamvPos, 10
        )

        self.wamv_roll = 0.0
        self.wamv_pitch = 0.0
        self.wamv_yaw = 0.0
        self.wamv_heading = 0.0
        self.imu_sub = self.create_subscription(
            Imu, "/wamv/sensors/imu/imu/data", self.wamvPose, 10
        )

        self.pinger_elevation = 0.0
        self.pinger_range = 0.0
        self.pinger_bearing = 0.0
        self.pinger_sub = self.create_subscription(
            ParamVec, "/wamv/pingers/pinger/range_bearing", self.pingerData, 10
        )

        self.acoustics_range = 0.0
        self.acoustics_bearing = 0.0
        self.acoustics_sub = self.create_subscription(
            ParamVec,
            "/wamv/sensors/acoustics/receiver/range_bearing",
            self.acousticsData,
            10,
        )

        self.image_frame = ""
        self.image_sub = self.create_subscription(
            Image,
            "/wamv/sensors/cameras/middle_left_camera/image_raw",
            self.imageCapture,
            10,
        )

        self.br = CvBridge()

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.goal_w = 0.0
        self.goal_sub = self.create_subscription(
            PoseStamped, "/vrx/stationkeeping/goal", self.goalPos, 10
        )

        self.numPoints = 0
        self.waypointArray = ""
        self.waypoint_sub = self.create_subscription(
            PoseArray, "/vrx/wayfinding/waypoints", self.waypointList, 10
        )

        # From discussions with Alex. The animal position is passed every 10 seconds and
        # in the same fromat as the Stationkeeping Goal, so we make three copies of the
        # goal and run from that until we figure out perception.

        self.animal0_x = 0.0
        self.animal0_y = 0.0
        self.animal0_z = 0.0
        self.animal0l_w = 0.0
        self.animal0_sub = self.create_subscription(
            PoseStamped, "/vrx/wildlife/animal0/pose", self.animal0Pos, 10
        )

        self.animal1_x = 0.0
        self.animal1_y = 0.0
        self.animal1_z = 0.0
        self.animal1l_w = 0.0
        self.animal1_sub = self.create_subscription(
            PoseStamped, "/vrx/wildlife/animal1/pose", self.animal1Pos, 10
        )

        self.animal2_x = 0.0
        self.animal2_y = 0.0
        self.animal2_z = 0.0
        self.animal2l_w = 0.0
        self.animal2_sub = self.create_subscription(
            PoseStamped, "/vrx/wildlife/animal0/pose", self.animal2Pos, 10
        )

        pub_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.right_thrust_pub = self.create_publisher(
            Float64, "/wamv/thrusters/right/thrust", qos_profile=pub_qos
        )
        self.middle_thrust_pub = self.create_publisher(
            Float64, "/wamv/thrusters/middle/thrust", qos_profile=pub_qos
        )
        self.left_thrust_pub = self.create_publisher(
            Float64, "/wamv/thrusters/left/thrust", qos_profile=pub_qos
        )
        self.thrust_msg = Float64()
        self.thrust_msg.data = 30.0
        self.pos_msg = Float64()
        self.pos_msg.data = 0.0
        self.loopCount = 0

        self.right_pos_pub = self.create_publisher(
            Float64, "/wamv/thrusters/right/pos", qos_profile=pub_qos
        )
        self.middle_pos_pub = self.create_publisher(
            Float64, "/wamv/thrusters/middle/pos", qos_profile=pub_qos
        )
        self.left_pos_pub = self.create_publisher(
            Float64, "/wamv/thrusters/left/pos", qos_profile=pub_qos
        )
        self.thrust_msg = Float64()
        self.thrust_msg.data = 30.0

        self.landmark_pub = self.create_publisher(
            PoseStamped, "/vrx/perception/landmark", qos_profile=pub_qos
        )
        self.landmark_msg = PoseStamped()
        self.landmark_msg.header.stamp = self.get_clock().now().to_msg()
        self.landmark_msg.header.frame_id = "mb_marker_buoy_red"
        self.landmark_msg.pose.position.x = -33.7227024
        self.landmark_msg.pose.position.y = 150.67402097
        self.landmark_msg.pose.position.z = 0.0

        self.wildlife_pub = self.create_publisher(
            PoseStamped, "/vrx/wildlife/animals", qos_profile=pub_qos
        )

        self.wildlife_msg = PoseStamped()
        self.wildlife_msg.header.stamp = self.get_clock().now().to_msg()
        self.wildlife_msg.header.frame_id = "platypus"
        self.wildlife_msg.pose.position.x = -33.7227024
        self.wildlife_msg.pose.position.y = 150.67402097
        self.wildlife_msg.pose.position.z = 0.0

        self.create_timer(0.5, self.sendCmds)

        self.bearing = 0.0
        self.distance = 0.0

        self.waypointArray_msg = PoseArray()

    def taskCB(self, msg):
        task_info = msg.params
        for p in task_info:
            if p.name == "name":
                self.task_type = p.value.string_value
            if p.name == "state":
                self.task_state = p.value.string_value

    def pingerData(self, msg):
        task_info = msg.params
        for p in task_info:
            if p.name == "elevation":
                self.pinger_elevation = p.value.string_value
            if p.name == "bearing":
                self.pinger_bearing = p.value.string_value
            if p.name == "range":
                self.pinger_range = p.value.string_value

    # Worked out by Michelle  Aguin-Lorenzana. Her idea is that this is the same as
    # repeated stationkeepinng maneuvers. You just have to make sure tht the bearing
    # to the target is updated. At the appropriate point in the code set goal.bearing
    # equal to self.acoustics_bearing and go for it.

    def acousticsData(self, msg):
        task_info = msg.params
        for p in task_info:
            if p.name == "bearing":
                self.acoustics_bearing = p.value.double_value
            if p.name == "range":
                self.acoustics_range = p.value._double_value

    def wamvPos(self, msg):
        self.wamv_latitude = msg.latitude
        self.wamv_longitude = msg.longitude
        self.wamv_altitude = msg.altitude

    def wamvPose(self, msg):
        # We learned this from Automatic Addison
        # Yaw is positive counterclockwise around the z axis, looking down.
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        self.wamv_roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        tmp = 2.0 * (w * y - z * x)
        if tmp > 1.0:
            tmp = 1.0
        elif tmp < -1.0:
            tmp = -1.0
        self.wamv_pitch = math.asin(tmp)

        self.wamv_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.wamv_heading = self.wamv_yaw

    def goalPos(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_z = msg.pose.position.z
        self.goal_w = msg.pose.orientation.w

    # From discussions with Alex. The animal position is passed every 10 seconds and
    # in the same fromat as the Stationkeeping Goal, so this is 3 copies of that listener
    # with the appropriate changes.

    def animal0Pos(self, msg):
        self.animal0_x = msg.pose.position.x
        self.animal0_y = msg.pose.position.y
        self.animal0_z = msg.pose.position.z
        self.animal0_w = msg.pose.orientation.w

    def animal1Pos(self, msg):
        self.animal1_x = msg.pose.position.x
        self.animal1_y = msg.pose.position.y
        self.animal1_z = msg.pose.position.z
        self.animal1_w = msg.pose.orientation.w

    def animal2Pos(self, msg):
        self.animal2_x = msg.pose.position.x
        self.animal2_y = msg.pose.position.y
        self.animal2_z = msg.pose.position.z
        self.animal2_w = msg.pose.orientation.w

    # Worked out by Christian Sopa, using his assistant, ChatGPT. ChatGPT came up with
    # the idea of iterating the PoseArray. We talked about where this could best be done
    # and decided, in the interest of time, to go with it right here and discuss the
    # process in more detail over the next month.

    def waypointList(self, msg):
        self.numPoints = 1
        self.waypointArray = []
        loc = (self.wamv_latitude, self.wamv_longitude, 0.0, 0.0)
        self.waypointArray.append(loc)
        for pose in msg.poses:
            position = pose.position
            loc = (position.x, position.y, 0.0, 0.0)
            self.waypointArray.append(loc)
            self.numPoints += 1

    # Sample code from Automatic Addsion's post
    # modified to suit VRX Application
    def imageCapture(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        self.get_logger().info("Receiving video frame")
        print("Receiving video frame")

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Display image
        cv2.imshow("camera", current_frame)

        cv2.waitKey(1)

    def getDistance(self, point1, point2):
        loc1 = (point1[0], point1[1])
        loc2 = (point2[0], point2[1])
        return hs.haversine(loc1, loc2, unit=Unit.METERS)

    def getBearing(self, point1, point2):
        theta_a = math.radians(point1[0])
        theta_b = math.radians(point2[0])
        X = math.cos(theta_b) * math.sin(math.radians(point2[1] - point1[1]))
        Y = math.cos(theta_a) * math.sin(theta_b) - math.sin(theta_a) * math.cos(
            theta_b
        ) * math.cos(math.radians(point2[1] - point1[1]))
        return math.atan2(X, Y)

    # https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/

    def computeCosts(self, waypointArray):
        V = self.numPoints
        rows, cols = (V, V)
        graph = [[0 for i in range(cols)] for j in range(rows)]

        for i in range(cols):
            for j in range(rows):
                loc1 = (waypointArray[i][0], waypointArray[i][1])
                loc2 = (waypointArray[j][0], waypointArray[j][1])
                graph[i][j] = self.getDistance(loc1, loc2)

        # This is equivalent to the following statements:

        # graph[0][0] = dist(waypoints[0], waypoints[0])
        # graph[0][1] = dist(waypoints[0], waypoints[1])
        # graph[0][2] = dist(waypoints[0], waypoints[2])
        # graph[0][3] = dist(waypoints[0], waypoints[3])
        # graph[1][0] = dist(waypoints[1], waypoints[0])
        # graph[1][1] = dist(waypoints[1], waypoints[1])
        # graph[1][2] = dist(waypoints[1], waypoints[2])
        # graph[1][3] = dist(waypoints[1], waypoints[3])
        # graph[2][0] = dist(waypoints[2], waypoints[0])
        # graph[2][1] = dist(waypoints[2], waypoints[1])
        # graph[2][2] = dist(waypoints[2], waypoints[2])
        # graph[2][3] = dist(waypoints[2], waypoints[3])
        # graph[3][0] = dist(waypoints[3], waypoints[0])
        # graph[3][1] = dist(waypoints[3], waypoints[1])
        # graph[3][2] = dist(waypoints[3], waypoints[2])
        # graph[3][3] = dist(waypoints[3], waypoints[3])

        return graph

    def waypointSelection(self, graph, s):
        # make an array to hold the waypoint identification
        vertex = []
        for i in range(self.numPoints):
            if i != s:
                vertex.append(i)

        # store minimum weight Hamiltonian Cycle
        min_path = maxsize
        best_path = ""
        next_permutation = permutations(vertex)
        for i in next_permutation:
            # store current Path weight(cost)
            current_pathweight = 0
            path = ""
            # compute current path weight
            k = s
            for j in i:
                print(k, j)
                output = "%r->%r" % (k, j)
                current_pathweight += graph[k][j]
                k = j
                path = "%s %s" % (path, output)

            # update minimum
            min_path = min(min_path, current_pathweight)
            if min_path == current_pathweight:
                best_path = path

        return best_path

    def steer(self, speed):
        # This is a crude example of feeding error back into the equation to
        # steer a vehicle. We don't know what the wind speed and direction are
        # during competition, so if we want to correct for them, we are going
        # to have to use vector math and know what we can expect under conditions
        # of no wind.
        error = self.bearing - (1.5708 - self.wamv_heading)
        if error < -0.5:
            error = -0.5
        elif error > 0.5:
            error = 0.5
        self.left = speed
        self.middle = 0.0
        self.right = speed
        self.pos_msg.data = error
        self.thrust_msg.data = self.left
        self.right_thrust_pub.publish(self.thrust_msg)
        self.right_pos_pub.publish(self.pos_msg)
        self.thrust_msg.data = self.middle
        self.middle_thrust_pub.publish(self.thrust_msg)
        self.thrust_msg.data = self.right
        self.left_thrust_pub.publish(self.thrust_msg)
        self.left_pos_pub.publish(self.pos_msg)
        steering_template = (
            "Heading: {!r} BeARING: {!r} Error: {!r} Left: {!r} Right: {!r}"
        )
        print(
            steering_template.format(
                self.wamv_heading * 180 / math.pi,
                self.bearing * 180 / math.pi,
                error,
                self.left,
                self.right,
            )
        )

    # def steer(self):
    #     #commented incase causes any errors;needs testing
    #     # Calculate position error (Euclidean distance)
    #     position_error = math.sqrt(
    #         (self.goal_x - self.wamv_latitude) ** 2 +
    #         (self.goal_y - self.wamv_longitude) ** 2
    #     )

    #     # Calculate heading error (positive difference in radians)
    #     heading_error = abs(self.goal_w - self.wamv_heading)

    #     # Calculate the total pose error using the given formula
    #     turn_thrust = 0.5*heading_error
    #     k = 0.75  # Weighting term guess
    #     pose_error = position_error + (k **position_error)*( heading_error) #based on document

    #     # Now, adjust thrust based on the pose error
    #     max_thrust = 100.0  # Define your maximum thrust value here
    #     min_thrust = 0.0    # Define your minimum thrust value here

    #     thrust_adjustment = pose_error*max_thrust

    #     # Limit thrust within the specified range
    #     thrust_adjusted = max(min_thrust, min(max_thrust, thrust_adjustment))
    #     #in theory the bottom segment will ensure it rotates first till it points to the goal, then drives to goal
    #     if !(0 <= heading_error <= 5):
    #         thrust_adjusted =0

    #     # This is a bit of a guess. To account for rotation i'm adding power to a specific motor based on
    #     # heading error(it may have to spin completly 360 to to orient itself)
    #     self.left_thrust_pub.publish(Float64(data=thrust_adjusted+turn_thrust))
    #     self.right_thrust_pub.publish(Float64(data=thrust_adjusted))

    def circleLeft(self):
        self.right = 60.0
        self.thrust_msg.data = self.right
        self.right_thrust_pub.publish(self.thrust_msg)
        self.left = 40.0
        self.thrust_msg.data = 40.0
        self.left_thrust_pub.publish(self.thrust_msg)
        self.loopCount += 1
        steering_template = "Heading: {!r} Left: {!r} Right: {!r}"
        print(
            steering_template.format(
                self.wamv_heading * 180 / math.pi, error, self.left, self.right
            )
        )

    def circleRight(self):
        self.right = 40.0
        self.thrust_msg.data = self.right
        self.right_thrust - pub.publish(slf.thrust_msg)
        self.left = 60.0
        self.thrust_msg.data = self.left
        self.left_thrust_pub.publish(self.thrust_msg)
        self.loopCount += 1
        steering_template = "Heading: {!r} Left: {!r} Right: {!r}"
        print(
            steering_template.format(
                self.wamv_heading * 180 / math.pi, error, self.left, self.right
            )
        )

    def moveForward(self):
        self.right_thrust_pub.publish(self.thrust_msg)
        self.left_thrust_pub.publish(self.thrust_msg)
        self.loopCount += 1

    def stop(self):
        self.thrust_msg.data = 0.0
        self.right_thrust_pub.publish(self.thrust_msg)
        self.left_thrust_pub.publish(self.thrust_msg)

    def publishBuoyLoc(self):
        if self.loopCount > 10:
            self.landmark_pub.publish(self.landmark_msg)
            self.loopCount = 0
        self.loopCount += 1

    def sendCmds(self):
        if rclpy.ok():
            match self.task_type:
                case "stationkeeping":
                    loc_template = "Lat: {!r} Lon: {!r}"
                    print(loc_template.format(self.wamv_latitude, self.wamv_longitude))
                    print(loc_template.format(self.goal_x, self.goal_y))
                    self.loc1 = (self.wamv_latitude, self.wamv_longitude)
                    self.loc2 = (self.goal_x, self.goal_y)
                    self.distance = self.getDistance(self.loc1, self.loc2)
                    self.bearing = self.getBearing(self.loc1, self.loc2)
                    course_template = "Bearing: {!r} Distance: {!r}"
                    print(
                        course_template.format(
                            self.bearing * 180 / math.pi, self.distance
                        )
                    )
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for stationkeeping task to start...")
                    elif self.task_state == "running":
                        if self.distance > 1.0:
                            print("Taking up Station.")
                            self.steer(30.0)
                        else:
                            print("Keeping Station.")
                            self.stop()
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                        rclpy.shutdown()
                case "wayfinding":
                    loc_template = "Lat: {!r} Lon: {!r}"
                    print(loc_template.format(self.wamv_latitude, self.wamv_longitude))
                    self.loc1 = (self.wamv_latitude, self.wamv_longitude)
                    if self.task_state == "initial":
                        print("Waiting for wayfinding task to start...")
                    elif self.task_state == "ready":
                        print("Computing path costs.")
                        new_path = self.waypointSelection(
                            self.computeCosts(self.waypointArray), 0
                        )
                        print(new_path)
                    elif self.task_state == "running":
                        self.steer(30.0)
                    elif self.task_state == "finished":
                        self.stop()
                        rclpy.shutdown()
                case "perception":
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for perception task to start...")
                    elif self.task_state == "running":
                        print("Perception task is running")
                        self.publishBuoyLoc()
                    elif self.task_state == "finished":
                        print("Task ended...")
                        rclpy.shutdown()
                case "acoustic_perception":
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for acoustic perception task to start...")
                    elif self.task_state == "running":
                        print("Waiting for acoustic perception task to complete...")
                        self.bearing = self.pinger_bearing
                        self.distance = self.pinger_range
                        if self.distance > 1.0:
                            print("Chasing the target.")
                            self.steer(30.0)
                        else:
                            print("On top of target.")
                            self.stop()
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                        rclpy.shutdown()

                # From discussions with Alex. The animal position is passed every 10 seconds and
                # in the same fromat as the Stationkeeping Goal, so we are going to set the new heading
                # and expected distance from those messages until we work out how perception can do
                # the job better.

                case "wildlife":
                    loc_template = "Lat: {!r} Lon: {!r}"
                    print(loc_template.format(self.wamv_latitude, self.wamv_longitude))
                    print(loc_template.format(self.goal_x, self.goal_y))
                    self.loc1 = (self.wamv_latitude, self.wamv_longitude)
                    course_template = "Chasing the Target. Bearing: {!r} Distance: {!r}"
                    print(
                        course_template.format(
                            self.bearing * 180 / math.pi, self.distance
                        )
                    )
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for wildlife task to start...")
                    elif self.task_state == "running":
                        if self.animal0_x < 0.0 and self.animal0_comp > 0:
                            self.loc2 = (self.animal0_x, self.animal0_y)
                            self.distance = self.getDistance(self.loc1, self.loc2)
                            self.bearing = self.getBearing(self.loc1, self.loc2)
                            if self.distance > 1.0:
                                print(
                                    course_template.format(
                                        self.bearing * 180 / math.pi, self.distance
                                    )
                                )
                                self.steer(30.0)
                            else:
                                print("Completed animal 0.")
                            self.stop()
                        if self.animal1_x < 0.0 and self.animal1_comp > 0:
                            self.loc2 = (self.animal1_x, self.animal1_y)
                            self.distance = self.getDistance(self.loc1, self.loc2)
                            self.bearing = self.getBearing(self.loc1, self.loc2)
                            if self.distance > 1.0:
                                print(
                                    course_template.format(
                                        self.bearing * 180 / math.pi, self.distance
                                    )
                                )
                                self.steer(30.0)
                            else:
                                print("Completed animal 1.")
                        if self.animal2_x < 0.0 and self.animal2_comp > 0:
                            self.loc2 = (self.animal2_x, self.animal1_y)
                            self.distance = self.getDistance(self.loc1, self.loc2)
                            self.bearing = self.getBearing(self.loc1, self.loc2)
                            if self.distance > 1.0:
                                print(
                                    course_template.format(
                                        self.bearing * 180 / math.pi, self.distance
                                    )
                                )
                                self.steer(30.0)
                            else:
                                print("Completed animal 2.")
                        print("Waiting for wildlife task to complete...")
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                case "follow_the_path":
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for follow the path task to start...")
                    elif self.task_state == "running":
                        print("Waiting for follow the path task to complete...")
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                case "acoustic_tracking":
                    self.distance = self.acoustics_range
                    self.bearing = self.acoustics_bearing
                    course_template = "Bearing: {!r} Distance: {!r}"
                    print(
                        course_template.format(
                            self.bearing * 180 / math.pi, self.distance
                        )
                    )
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for Acoustics Tracking task to start...")
                    elif self.task_state == "running":
                        if self.distance > 1.0:
                            print("Following Beacon.")
                            self.steer(30.0)
                        else:
                            print("Found Beacon.")
                            self.stop()
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                        rclpy.shutdown()
                case "scan_and_dock":
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for scan dock and deliver task to start...")
                    elif self.task_state == "running":
                        print("Waiting for scan dock and deliver task to complete...")
                    elif self.task_state == "finished":
                        self.stop()
                        print("Task ended...")
                case _:
                    print(self.task_state)
                    if self.task_state == ("initial" or "ready"):
                        print("Waiting for default task to start...")
                    elif self.task_state == "running":
                        self.right_thrust_pub.publish(self.thrust_msg)
                        self.left_thrust_pub.publish(self.thrust_msg)
                    elif self.task_state == "finished":
                        print("Task ended...")
                        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    walloping_window_blind = WallopingWindowBlind()
    rclpy.spin(walloping_window_blind)


if __name__ == "__main__":
    main()
