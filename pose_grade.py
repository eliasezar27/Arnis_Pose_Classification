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


# Ground Truths for each strikes
def strike_grade(part_line, bboxList, key):
	grade = 0
	kp_shown = 0 < [i in part_line for i in range(0, 34)].count(True) < 34
	point_baston = 0

	if kp_shown:

		angle_dict = joint_angles(part_line, bboxList)
		# print(angle_dict)

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

		# Striking Technique ------------------------------------------------------------------------------------>

		# Pugay
		if key == 1:
			overall = 6

			if r_elbow in range(50, 90):
				grade += 1

			if l_elbow >= 130:
				grade += 1

			if r_shoulder in range(75, 100):
				grade += 1

			if l_shoulder in range(91, 115):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Handa
		elif key == 2:
			overall = 6

			if r_elbow >= 140:
				grade += 1

			if l_elbow >= 140:
				grade += 1

			if r_shoulder in range(91, 121):
				grade += 1

			if l_shoulder in range(91, 121):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Temple
		elif key == 3:
			overall = 9

			if r_elbow > 10:
				grade += 1

			if r_shoulder > 90:
				grade += 1

			if l_elbow < 90:
				grade += 1

			if l_shoulder in range(70, 150):
				grade += 1

			if r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			if bbox_index == 2:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Temple
		elif key == 4:
			overall = 9

			if r_elbow > 10:
				grade += 1

			if r_shoulder > 90:
				grade += 1

			if l_elbow < 90:
				grade += 1

			if l_shoulder in range(70, 150):
				grade += 1

			if r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			if bbox_index == 1:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Shoulder
		elif key == 5:
			overall = 9

			if r_elbow > 10:
				grade += 1

			if r_shoulder > 90:
				grade += 1

			if l_elbow < 90:
				grade += 1

			if l_shoulder in range(70, 150):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			if bbox_index == 2:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Shoulder
		elif key == 6:
			overall = 9

			if r_elbow > 10:
				grade += 1

			if r_shoulder > 90:
				grade += 1

			if l_elbow < 90:
				grade += 1

			if l_shoulder in range(70, 150):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			if bbox_index == 1:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Stomach Thrust
		elif key == 7:
			overall = 8

			if r_elbow > 135:
				grade += 1

			if r_shoulder in range(80, 110):
				grade += 1

			if l_elbow in range(80, 110):
				grade += 1

			if l_shoulder in range(80, 110):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Chest Thrust
		elif key == 8:
			overall = 8

			if r_elbow in range(90, 160):
				grade += 1

			if r_shoulder in range(100, 160):
				grade += 1

			if l_elbow in range(90, 160):
				grade += 1

			if l_shoulder in range(10, 90):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Chest Thrust
		elif key == 9:
			overall = 8

			if r_elbow in range(45, 135):
				grade += 1

			if r_shoulder in range(25, 90):
				grade += 1

			if l_elbow == 0:
				grade += 1

			if l_shoulder == 0:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Leg Strike
		elif key == 10:
			overall = 8

			if r_elbow > 160:
				grade += 1

			if r_shoulder in range(90, 110):
				grade += 1

			if l_elbow in range(35, 90):
				grade += 1

			if l_shoulder in range(45, 135):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Leg Strike
		elif key == 11:
			overall = 8

			if r_elbow > 160:
				grade += 1

			if r_shoulder in range(50, 90):
				grade += 1

			if l_elbow in range(35, 90):
				grade += 1

			if l_shoulder in range(45, 135):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Eye Thrust
		elif key == 12:
			overall = 8

			if r_elbow in range(90, 160):
				grade += 1

			if r_shoulder > 135:
				grade += 1

			if l_elbow in range(90, 160):
				grade += 1

			if l_shoulder in range(10, 90):
				grade += 1

			if r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Eye Thrust
		elif key == 13:
			overall = 8

			if r_elbow in range(45, 135):
				grade += 1

			if r_shoulder in range(25, 90):
				grade += 1

			if l_elbow == 0:
				grade += 1

			if l_shoulder == 0:
				grade += 1

			if r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Crown Strike
		elif key == 14:
			overall = 6

			if r_elbow > 120:
				grade += 1

			if r_shoulder in range(60, 120):
				grade += 1

			if l_elbow in range(60, 120):
				grade += 1

			if l_shoulder in range(60, 120):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# BLOCKING TECHNIQUES ------------------------------------------------------------------------------------>

		# Left Temple Block
		elif key == 15:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(43, 85):
				grade += 1

			if l_elbow > 100:
				grade += 1

			if l_shoulder > 100:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Temple Block
		elif key == 16:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(107, 136):
				grade += 1

			if l_elbow > 130:
				grade += 1

			if l_shoulder < 90:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Shoulder Block
		elif key == 17:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(43, 85):
				grade += 1

			if l_elbow > 100:
				grade += 1

			if l_shoulder > 100:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# elif r_elbow > 140 and r_shoulder in range(75, 90) and l_elbow > 140 and l_shoulder > 150 and not r_wrist_up and l_wrist_up and not rw_near_rs and not lw_near_ls:

		# Right Shoulder Block
		elif key == 18:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(107, 136):
				grade += 1

			if l_elbow > 130:
				grade += 1

			if l_shoulder < 90:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Stomach Thrust Block
		elif key == 19:
			overall = 8

			if r_elbow > 135:
				grade += 1

			if r_shoulder < 20:
				grade += 1

			if l_elbow > 160:
				grade += 1

			if l_shoulder in range(125, 145):
				grade += 1

			if r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Chest Block
		elif key == 20:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(43, 85):
				grade += 1

			if l_elbow > 100:
				grade += 1

			if l_shoulder > 100:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Chest Block
		elif key == 21:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(107, 136):
				grade += 1

			if l_elbow > 130:
				grade += 1

			if l_shoulder < 90:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Leg Block
		elif key == 22:
			overall = 8

			if r_elbow > 160:
				grade += 1

			if r_shoulder > 100:
				grade += 1

			if l_elbow in range(0, 50):
				grade += 1

			if l_shoulder in range(92, 110):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Leg Block
		elif key == 23:
			overall = 8

			if r_elbow > 100:
				grade += 1

			if r_shoulder < 90:
				grade += 1

			if l_elbow in range(0, 50):
				grade += 1

			if l_shoulder in range(92, 110):
				grade += 1

			if not r_wrist_up:
				grade += 1

			if not l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Left Eye Block
		elif key == 24:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(43, 85):
				grade += 1

			if l_elbow > 100:
				grade += 1

			if l_shoulder > 100:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if not rw_near_rs:
				grade += 1

			if lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Right Eye Block
		elif key == 25:
			overall = 8

			if r_elbow > 130:
				grade += 1

			if r_shoulder in range(107, 136):
				grade += 1

			if l_elbow > 130:
				grade += 1

			if l_shoulder < 90:
				grade += 1

			if not r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			if rw_near_rs:
				grade += 1

			if not lw_near_ls:
				grade += 1

			grade = round((grade / overall) * 100, 2)

		# Rising Block
		elif key == 26:
			overall = 6

			if r_elbow > 150:
				grade += 1

			if r_shoulder in range(80, 160):
				grade += 1

			if l_elbow in range(80, 110):
				grade += 1

			if l_shoulder > 140:
				grade += 1

			if r_wrist_up:
				grade += 1

			if l_wrist_up:
				grade += 1

			grade = round((grade / overall) * 100, 2)

	return grade, point_baston


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
