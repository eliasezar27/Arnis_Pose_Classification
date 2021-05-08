import math


def euclidean(point1, point2):
	return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_calc(p0, p1, p2):
	# p1 is center point from where we measured angle between p0 and p2

	try:
		a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
		b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
		c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
		angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
	except:
		return 0

	return int(angle)


# Ground Truth before the actual strike pose in case of movement classification

# def prestrike(part_line, bboxList):
# 	activate = -1
# 	kp_shown = 0 < [i in part_line for i in range(0, 34)].count(True) < 34
#
# 	if kp_shown:
#
# 		angle_dict = joint_angles(part_line, bboxList)
# 		print(angle_dict)
#
# 		r_elbow = angle_dict['right elbow']
# 		l_elbow = angle_dict['left elbow']
# 		r_shoulder = angle_dict['right shoulder']
# 		l_shoulder = angle_dict['left shoulder']
# 		r_hip = angle_dict['right hip']
# 		l_hip = angle_dict['left hip']
# 		r_knee = angle_dict['right knee']
# 		l_knee = angle_dict['left knee']
# 		r_wrist_up = angle_dict['r_wrist_raised']
# 		l_wrist_up = angle_dict['l_wrist_raised']
# 		r_elbow_up = angle_dict['r_elbow_raised']
# 		l_elbow_up = angle_dict['l_elbow_raised']
# 		rw_near_rs = angle_dict['rw_near_rs']
# 		lw_near_ls = angle_dict['lw_near_ls']
# 		bbox_index = angle_dict['index_bbox'] # pt of the bbox near the right wrist
#
# 		# Pugay
# 		if r_elbow in range(50, 90) and l_elbow >= 130 and r_shoulder in range(75, 100) and l_shoulder in range(91, 115) and not r_wrist_up and not l_wrist_up:
# 			activate = 0.2
# 			label = "PUGAY"
# 			print(label)
#
# 		# Handa
# 		elif r_elbow >= 155 and l_elbow >= 155 and r_shoulder in range(91, 121) and l_shoulder in range(91, 121) and not r_wrist_up and not l_wrist_up:
# 			activate = 0.1
# 			label = "HANDA"
# 			print(label)
#
# 		# Left Temple
# 		elif r_elbow in range(80, 91) and r_shoulder >= 160 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 1
# 			label = "Left Temple pre-Strike"
# 			print(label)
#
# 		# Right Temple
# 		elif r_elbow <= 90 and r_shoulder < 135 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
# 			activate = 2
# 			label = "Right Temple pre-Strike"
# 			print(label)
#
# 		# Left Shoulder same pre strike as left temple
# 		elif r_elbow in range(80, 91) and r_shoulder >= 160 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 3
# 			label = "Left Shoulder pre-Strike"
# 			print(label)
#
# 		# Right Shoulder same pre strike as right temple
# 		elif r_elbow <= 90 and r_shoulder < 135 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
# 			activate = 4
# 			label = "Right Shoulder pre-Strike"
# 			print(label)
#
# 		# Stomach Thrust
# 		elif r_elbow > 135 and r_shoulder in range(90,146) and l_elbow > 10 and l_shoulder > 10 and not r_wrist_up and l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 5
# 			label = "Stomach Pre-thrust"
# 			print(label)
#
# 		# Left Chest Thrust
# 		elif r_elbow <= 90 and r_shoulder > 135 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 6
# 			label = "Left Chest Pre-thrust"
# 			print(label)
#
# 		# Right Chest Thrust
# 		elif r_elbow > 150 and r_shoulder in range(25, 91) and l_elbow in range(45, 135) and l_shoulder > 150 and not r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
# 			activate = 7
# 			label = "Right Chest Pre-thrust"
# 			print(label)
#
# 		# Right Leg Strike
# 		elif r_elbow <= 90 and r_shoulder < 135 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and not l_wrist_up and not rw_near_rs and lw_near_ls:
# 			activate = 8
# 			label = "Right Leg Pre-strike"
# 			print(label)
#
# 		# Left Leg Strike
# 		elif r_elbow in range(80, 91) and r_shoulder >= 160 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and not l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 9
# 			label = "Left Leg Pre-strike"
# 			print(label)
#
# 		# Left Eye Thrust
# 		elif r_elbow <= 90 and r_shoulder > 135 and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and rw_near_rs and lw_near_ls:
# 			activate = 10
# 			label = "Left Eye Pre-thrust"
# 			print(label)
#
# 		# Right Eye Thrust
# 		elif r_elbow > 150 and r_shoulder in range(25, 91) and l_elbow in range(45, 135) and l_shoulder > 150 and not r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
# 			activate = 11
# 			label = "Right Eye Pre-thrust"
# 			print(label)
#
# 		# Crown Strike
# 		elif r_elbow in range(70, 120) and r_shoulder in range(110, 160) and l_elbow > 10 and l_shoulder > 10 and r_wrist_up and l_wrist_up and lw_near_ls and r_elbow_up:
# 			activate = 12
# 			label = "Crown Pre-strike"
# 			print(label)
#
# 	else:
# 		activate = -2
#
# 	return activate


# Ground Truths for each strikes
def strike(part_line, bboxList):
	label = ''
	kp_shown = 0 < [i in part_line for i in range(0, 34)].count(True) < 34
	point_baston = 0

	if kp_shown:

		angle_dict = joint_angles(part_line, bboxList)
		print(angle_dict)

		r_elbow = angle_dict['right elbow']
		l_elbow = angle_dict['left elbow']
		r_shoulder = angle_dict['right shoulder']
		l_shoulder = angle_dict['left shoulder']
		# r_hip = angle_dict['right hip']
		# l_hip = angle_dict['left hip']
		# r_knee = angle_dict['right knee']
		# l_knee = angle_dict['left knee']
		r_wrist_up = angle_dict['r_wrist_raised']
		l_wrist_up = angle_dict['l_wrist_raised']
		# r_elbow_up = angle_dict['r_elbow_raised']
		# l_elbow_up = angle_dict['l_elbow_raised']
		rw_near_rs = angle_dict['rw_near_rs']
		lw_near_ls = angle_dict['lw_near_ls']

		# pt of the bbox near the right wrist
		bbox_index = angle_dict['index_bbox']
		# pt of the bbox with red circle
		point_baston = angle_dict['draw_circle']

# Striking Technique

		# Pugay
		if r_elbow in range(50, 90) and l_elbow >= 130 and r_shoulder in range(75, 100) and l_shoulder in range(91, 115) and not r_wrist_up and not l_wrist_up:
			label = "PUGAY"
			print(label)

		# Handa
		elif r_elbow >= 140 and l_elbow >= 140 and r_shoulder in range(91, 121) and l_shoulder in range(91, 121) and not r_wrist_up and not l_wrist_up:
			label = "HANDA"
			print(label)

		# Left Temple
		elif r_elbow > 10 and r_shoulder > 90 and l_elbow < 90 and l_shoulder in range(70, 150) and r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls and bbox_index == 2:
			label = "Left Temple Strike"
			print(label)

		# Right Temple
		elif r_elbow > 10 and r_shoulder > 90 and l_elbow < 90 and l_shoulder in range(70, 150) and r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls and bbox_index == 1:
			label = "Right Temple Strike"
			print(label)

		# Left Shoulder
		elif r_elbow > 10 and r_shoulder > 90 and l_elbow < 90 and l_shoulder in range(70, 150) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls and bbox_index == 2:
			label = "Left Shoulder Strike"
			print(label)

		# Right Shoulder
		elif r_elbow > 10 and r_shoulder > 90 and l_elbow < 90 and l_shoulder in range(70, 150) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls and bbox_index == 1:
			label = "Right Shoulder Strike"
			print(label)

		# Stomach Thrust
		elif r_elbow > 135 and r_shoulder in range(80, 110) and l_elbow in range(80, 110) and l_shoulder in range(80, 110) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Stomach Thrust"
			print(label)

		# Left Chest Thrust
		elif r_elbow in range(90, 160) and r_shoulder in range(100, 160) and l_elbow in range(90, 160) and l_shoulder in range(10, 90) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Left Chest Thrust"
			print(label)

		# Right Chest Thrust
		elif r_elbow in range(45, 135) and r_shoulder in range(25, 90) and l_elbow == 0 and l_shoulder == 0 and not r_wrist_up and not l_wrist_up and not rw_near_rs and not lw_near_ls:
			label = "Right Chest Thrust"
			print(label)

		# Right Leg Strike
		elif r_elbow > 160 and r_shoulder in range(90, 110) and l_elbow in range(35, 90) and l_shoulder in range(45, 135) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Right Leg Strike"
			print(label)

		# Left Leg Strike
		elif r_elbow > 160 and r_shoulder in range(50, 90) and l_elbow in range(35, 90) and l_shoulder in range(45, 135) and not r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Left Leg Strike"
			print(label)

		# Left Eye Thrust
		elif r_elbow in range(90, 160) and r_shoulder > 135 and l_elbow in range(90, 160) and l_shoulder in range(10, 90) and r_wrist_up and not l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Left Eye Thrust"
			print(label)

		# Right Eye Thrust
		elif r_elbow in range(45, 135) and r_shoulder in range(25, 90) and l_elbow == 0 and l_shoulder == 0 and r_wrist_up and not l_wrist_up and not rw_near_rs and not lw_near_ls:
			label = "Right Eye Thrust"
			print(label)

		# Crown Strike
		elif r_elbow > 120 and r_shoulder in range(60, 120) and l_elbow in range(60, 120) and l_shoulder in range(60, 120) and not r_wrist_up and not l_wrist_up:
			label = "Crown Strike"
			print(label)

# BLOCKING TECHNIQUES

		# Left Temple Block
		elif r_elbow > 130 and r_shoulder in range(43, 63) and l_elbow > 100 and l_shoulder > 100 and not r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
			label = "Left Temple Block"
			print(label)

		# Right Block
		elif r_elbow > 130 and r_shoulder in range(148, 158) and l_elbow > 150 and l_shoulder in range(30, 50) and not r_wrist_up and l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Right Temple Block"
			print(label)

		# Left Shoulder Block
		elif r_elbow > 140 and r_shoulder in range(75, 90) and l_elbow > 140 and l_shoulder > 150 and not r_wrist_up and l_wrist_up and not rw_near_rs and not lw_near_ls:
			label = "Left Shoulder Block"
			print(label)

		# Right Shoulder Block
		elif r_elbow > 180 and r_shoulder in range(115, 135) and l_elbow > 140 and l_shoulder < 20 and not r_wrist_up and l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Right Shoulder Block"
			print(label)

		# Stomach Thrust Block
		elif r_elbow > 135 and r_shoulder < 20 and l_elbow > 160 and l_shoulder in range(125, 145) and r_wrist_up and not l_wrist_up and not rw_near_rs and lw_near_ls:
			label = "Stomach Thrust Block"
			print(label)

		# Left Chest Block
		elif r_elbow > 140 and r_shoulder in range(65, 85) and l_elbow > 100 and l_shoulder > 140 and not r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
			label = "Left Chest Block"
			print(label)

		# Right Chest Block
		elif r_elbow > 160 and r_shoulder in range(109, 129) and l_elbow > 160 and l_shoulder < 50 and not r_wrist_up and l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Right Chest Block"
			print(label)

		# Right Leg Block
		elif r_elbow > 160 and r_shoulder > 100 and l_elbow in range(0, 50) and l_shoulder in range(92, 110) and not r_wrist_up and not l_wrist_up and rw_near_rs and lw_near_ls:
			label = "Right Leg Block"
			print(label)

		# Left Leg Block
		elif r_elbow > 100 and r_shoulder < 90 and l_elbow in range(0, 50) and l_shoulder in range(92, 110) and not r_wrist_up and not l_wrist_up and not rw_near_rs and lw_near_ls:
			label = "Left Leg Block"
			print(label)

		# Left Eye Block
		elif r_elbow in range(125, 145) and r_shoulder in range(54, 74) and l_elbow > 130 and l_shoulder > 150 and not r_wrist_up and l_wrist_up and not rw_near_rs and lw_near_ls:
			label = "Left Eye Block"
			print(label)

		# Right Eye Block
		elif r_elbow in range(106, 126) and r_shoulder in range(107, 127) and l_elbow > 130 and l_shoulder < 20 and not r_wrist_up and l_wrist_up and rw_near_rs and not lw_near_ls:
			label = "Right Eye Block"
			print(label)

		# Rising Block
		elif r_elbow > 150 and r_shoulder in range(120, 140) and l_elbow in range(80, 110) and l_shoulder > 160 and r_wrist_up and l_wrist_up:
			label = "Rising Block"
			print(label)

	else:
		label = 'Cannot find body'

	return label, point_baston


def joint_angles(part_line, bboxList):
	r_elbow = 0
	l_elbow = 0
	r_shoulder = 0
	l_shoulder = 0
	r_hip = 0
	l_hip = 0
	r_knee = 0
	l_knee = 0
	r_wrist_up = False
	l_wrist_up = False
	r_elbow_up = False
	l_elbow_up = False
	r_wrist_near_r_shoulder = False
	l_wrist_near_l_shoulder = False
	idx_bbox = -1
	draw_cir_idx = [2, 3, 0, 1]
	cir_idx = 2

	if 12 in part_line and 14 in part_line and 16 in part_line:
		r_elbow = angle_calc(part_line[12], part_line[14], part_line[16])
		rw_to_pt1 = euclidean(part_line[16], bboxList[0])
		rw_to_pt2 = euclidean(part_line[16], bboxList[1])
		rw_to_pt3 = euclidean(part_line[16], bboxList[2])
		rw_to_pt4 = euclidean(part_line[16], bboxList[3])

		if part_line[12][1] > part_line[16][1]:
			r_wrist_up = True

		if part_line[12][1] > part_line[14][1]:
			r_elbow_up = True

		rw_to_bbox = [rw_to_pt1, rw_to_pt2, rw_to_pt3, rw_to_pt4]
		min_rw_to_bbox = min(rw_to_bbox)
		idx_bbox = rw_to_bbox.index(min_rw_to_bbox)
		cir_idx = draw_cir_idx[idx_bbox]

	if 11 in part_line and 13 in part_line and 15 in part_line:
		l_elbow = angle_calc(part_line[11], part_line[13], part_line[15])

		if part_line[11][1] > part_line[15][1]:
			l_wrist_up = True

		if part_line[11][1] > part_line[13][1]:
			l_elbow_up = True

	if 11 in part_line and 12 in part_line and 14 in part_line:
		r_shoulder = angle_calc(part_line[11], part_line[12], part_line[14])

	if 11 in part_line and 12 in part_line and 20 in part_line:
		r_wrist_to_r_shoulder = euclidean(part_line[20], part_line[12])
		r_wrist_to_l_shoulder = euclidean(part_line[20], part_line[11])
		if r_wrist_to_r_shoulder < r_wrist_to_l_shoulder:
			r_wrist_near_r_shoulder = True

	if 11 in part_line and 12 in part_line and 19 in part_line:
		l_wrist_to_r_shoulder = euclidean(part_line[19], part_line[12])
		l_wrist_to_l_shoulder = euclidean(part_line[19], part_line[11])
		if l_wrist_to_l_shoulder < l_wrist_to_r_shoulder:
			l_wrist_near_l_shoulder = True

	if 12 in part_line and 11 in part_line and 13 in part_line:
		l_shoulder = angle_calc(part_line[12], part_line[11], part_line[13])

	if 26 in part_line and 24 in part_line and 23 in part_line:
		r_hip = angle_calc(part_line[26], part_line[24], part_line[23])

	if 25 in part_line and 23 in part_line and 24 in part_line:
		l_hip = angle_calc(part_line[25], part_line[23], part_line[24])

	if 24 in part_line and 26 in part_line and 28 in part_line:
		r_knee = angle_calc(part_line[24], part_line[26], part_line[28])

	if 23 in part_line and 25 in part_line and 27 in part_line:
		l_knee = angle_calc(part_line[23], part_line[25], part_line[27])

	angle_dict = {
					'right elbow': r_elbow, 'left elbow': l_elbow,
				 	'right shoulder': r_shoulder, 'left shoulder': l_shoulder,
				  	'right hip': r_hip, 'left hip': l_hip,
					'right knee': r_knee, 'left knee': l_knee,
					'r_wrist_raised': r_wrist_up, 'l_wrist_raised': l_wrist_up,
					'r_elbow_raised': r_elbow_up, 'l_elbow_raised': l_elbow_up,
					'rw_near_rs': r_wrist_near_r_shoulder, 'lw_near_ls': l_wrist_near_l_shoulder,
					'index_bbox': idx_bbox, 'draw_circle': cir_idx
				  }

	return angle_dict
