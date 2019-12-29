import numpy as np
import cv2 as cv


def Convert2DPointsTo3DPoints(points2D_L, points2D_R, E, P, points3D):

	# Details in:
	# D. Kurtagić, Trodimenzionalna rekonstrukcija scene iz dvije slike (završni rad - preddiplomski
	# studij). Elektrotehnički fakultet, Osijek, September 9, 2010.


    # ////// CONVERTED TO PYTHON //////


	# //Find the inverse of P (projection matrix)
	Pinv = []

	Pinv=np.linalg.pinv(P)

	# //Determine the singular value decomposition (svd) of E (essential matrix)
	U = np.zeros((3, 3))
	V = np.zeros((3, 3))
	Vt = np.zeros((3, 3))
	D = np.zeros((3, 3))
	cv.SVDecomp(E, D, U, Vt)


	# //Define W
	W = np.zeros((3, 3))
	W[0][1] = -1
	W[1][0] = 1
	W[2][2] = 1

	A = np.matmul(np.matmul(U, W), Vt)

	b = np.zeros((3, 1))
	b[0][0] = U[0][2]
	b[1][0] = U[1][2]
	b[2][0] = U[2][2]

	Ainv = np.zeros((3, 3))
	Ainv_b = np.zeros((3, 1))
	Ainv = np.linalg.pinv(A)
	Ainv_b = np.matmul(Ainv, b)

	Lpi = np.zeros((3, 1))
	Rpi = np.zeros((3, 1))
	ARpi = np.zeros((3, 1))

	S = np.zeros((2, 1))

	X = np.zeros((2, 2))
	x1 = []
	x2 = []
	x4 = []

	Y = np.zeros((2, 1))
	y1 = []
	y2 = []

	# //2D points in left (model) and right (scene) images
	Lm = np.zeros((3, 1))
	Rm = np.zeros((3, 1))



	# //Iteratively convert 2D point pairs to 3D points
	i=0
	for x, y in points2D_L:
		Lm[0][0] = x
		Lm[1][0] = y
		Lm[2][0] = 1

		Rm[0][0] = points2D_R[i][0]
		Rm[1][0] = points2D_R[i][1]
		Rm[2][0] = 1

		Lpi = np.matmul(Pinv, Lm)
		Rpi = np.matmul(Pinv, Rm)

		ARpi = np.matmul(Ainv, Rpi)
		x1 = np.matmul(np.transpose(Lpi), Lpi)
		x2 = np.matmul(np.transpose(Lpi), ARpi)
		x4 = np.matmul(np.transpose(ARpi), ARpi)

		X[0][0] = -x1[0][0]
		X[0][1] = x2[0][0]
		X[1][0] = x2[0][0]
		X[1][1] = -x4[0][0]

		y1 = np.matmul(np.transpose(Lpi), Ainv_b)
		y2 = np.matmul(np.transpose(ARpi), Ainv_b)

		Y[0][0] = -y1[0][0]
		Y[1][0] = y2[0][0]

		cv.solve(X, Y, S)

		s = S[0][0]
		t = S[1][0]

		Lpi[0][0] = s * Lpi[0][0]
		Lpi[1][0] = s * Lpi[1][0]
		Lpi[2][0] = s * Lpi[2][0]

		ARpi[0][0] = t * ARpi[0][0]
		ARpi[1][0] = t * ARpi[1][0]
		ARpi[2][0] = t * ARpi[2][0]

		points3D[0][i] = (Lpi[0][0] + ARpi[0][0] - Ainv_b[0][0]) / 2
		points3D[1][i] = (Lpi[1][0] + ARpi[1][0] - Ainv_b[1][0]) / 2
		points3D[2][i] = (Lpi[2][0] + ARpi[2][0] - Ainv_b[2][0]) / 2

		i += 1




