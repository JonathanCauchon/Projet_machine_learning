# Simplified model for Gif-7005 course project

from Modules import *

def clc():
	print ("\n"*10)


class ChirpedContraDC():
	def __init__(self, N = 1000, period = 322e-9, a = 10, kappa_max = 48000, kappa_min = 0, T = 300, \
		resolution = 1001, N_seg = 101, wvl_range = [1500e-9,1600e-9], central_wvl = 1550e-9, \
		alpha = 10, apod_shape = "gaussian"):

		# Class attributes
		self.N           =  N           #  int    Number of grating periods      [-]
		self.period      =  period      #  float  Period of the grating          [m]
		self.a           =  a           #  int    Apodization Gaussian constant  [-]
		self.kappa_max   =  kappa_max       #  float  Maximum coupling power         [m^-1]
		self.T           =  T           #  float  Device temperature             [K]
		self.resolution  =  resolution  #  int    Nb. of freq. points computed   [-]
		self.N_seg       =  N_seg       #  int    Nb. of apod. & chirp segments  [-]
		self.alpha       =  alpha       #  float  Propagation loss grating       [dB/cm]
		self.wvl_range   =  wvl_range   #  list   Start and end wavelengths      [m]
		self.kappa_min   =  kappa_min
		self.apod_shape  = apod_shape
		self.central_wvl = central_wvl
		# Note that gap is set to 100 nm

		# Constants
		self._antiRefCoeff = 0.01

		# Properties that will be set through methods
		self.apod_profile = None
		self.period_profile = None



	# Property functions: changing one property automatically affects others
	@ property
	def wavelength(self):
		return np.linspace(self.wvl_range[0], self.wvl_range[1], self.resolution)

	@ property
	def c(self):
		return 299792458


	# linear algebra numpy manipulation functions
	def switchTop(self, P):
		P_FF = np.asarray([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
		P_FG = np.asarray([[P[0][2],P[0][3]],[P[1][2],P[1][3]]])
		P_GF = np.asarray([[P[2][0],P[2][1]],[P[3][0],P[3][1]]])
		P_GG = np.asarray([[P[2][2],P[2][3]],[P[3][2],P[3][3]]])

		H1 = P_FF-np.matmul(np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1)),P_GF)
		H2 = np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1))
		H3 = np.matmul(-np.linalg.matrix_power(P_GG,-1),P_GF)
		H4 = np.linalg.matrix_power(P_GG,-1)
		H = np.vstack((np.hstack((H1,H2)),np.hstack((H3,H4))))

		return H

	# Swap columns of a given array
	def swap_cols(self, arr, frm, to):
		arr[:,[frm, to]] = arr[:,[to, frm]]
		return arr

	# Swap rows of a given array
	def swap_rows(self, arr, frm, to):
		arr[[frm, to],:] = arr[[to, frm],:]
		return arr
	    

	# Print iterations progress
	def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
		# Print New Line on Complete
		if iteration == total: 
		    print()


	def getPropConstants(self, plot=False):
		
		T0 = 300
		dneffdT = 1.87e-4      #[/K] assuming dneff/dn=1 (well confined mode)
		neffThermal = dneffdT*(self.T-T0)

		n1_1550 = 2.6
		n2_1550 = 2.4
		dn1 = -1.0e6
		dn2 = -1.1e6
		centralWL=1550e-9
		period=1550e-9/2.5/2 

		self.n1 = neffThermal + dn1*(self.wavelength - centralWL) + n1_1550
		self.n2 = neffThermal + dn2*(self.wavelength - centralWL) + n2_1550

		self.beta1 = 2*math.pi / self.wavelength * self.n1
		self.beta2 = 2*math.pi / self.wavelength * self.n2


	def getApodProfile(self, plot=False):
		segments = np.arange(0, self.N_seg)

		if self.apod_shape is "gaussian":
			self.apod_profile = np.exp(-self.a*(segments-self.N_seg/2)**2/self.N_seg**2)
			self.apod_profile -= self.apod_profile.min()
			self.apod_profile /= self.apod_profile.max()
			self.apod_profile *= self.kappa_max

		elif self.apod_shape is "tanh":
			alpha, beta = 2, 3
			apod = 1/2 * (1 + np.tanh(beta*(1-2*abs(2*segments/self.N_seg)**alpha)))
			apod = np.append(np.flip(apod[0:int(apod.size/2)]), apod[0:int(apod.size/2)])
			apod *= self.kappa
			self.apod_profile = apod

		if plot:
			plt.plot(self.apod_profile,"o")
			plt.show()

	def getChirpProfile(self, plot=False):

		# Period chirp
		if isinstance(self.period, list):
			self.period_profile = np.linspace(self.period[0], self.period[-1], self.N_seg)

		else:
			self.period_profile = self.period*np.ones(self.N_seg)

		if plot:
			plt.plot(self.period_profile,"o")
			plt.show()




	def propagate(self, bar):
		# initiate arrays
		T = np.zeros((1, self.resolution),dtype=complex)
		R = np.zeros((1, self.resolution),dtype=complex)
		T_co = np.zeros((1, self.resolution),dtype=complex)
		R_co = np.zeros((1, self.resolution),dtype=complex)
		
		E_Thru = np.zeros((1, self.resolution),dtype=complex)
		E_Drop = np.zeros((1, self.resolution),dtype=complex)

		LeftRightTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)
		TopDownTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)
		InOutTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)

		# kappa_apod = self.getApodProfile()
		kappa_apod = self.apod_profile

		mode_kappa_a1=1
		mode_kappa_a2=0 #no initial cross coupling
		mode_kappa_b2=1
		mode_kappa_b1=0

		j = cmath.sqrt(-1)      # imaginary

		alpha_e = 100*self.alpha/10*math.log(10)

		if bar:
			progressbar_width = self.resolution
			# Initial call to print 0% progress
			self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
		        
		for ii in range(self.resolution):
			if bar:
				clc()
				print("Propagating along grating...")
				self.printProgressBar(ii + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

			l_0 = 0
			for n in range(self.N_seg):

				l_seg = self.N/self.N_seg * self.period_profile[n]			

				kappa_12 = self.apod_profile[n]
				kappa_21 = np.conj(kappa_12);
				kappa_11 = self._antiRefCoeff * self.apod_profile[n]
				kappa_22 = self._antiRefCoeff * self.apod_profile[n]

				beta_del_1 = self.beta1[ii] - math.pi/self.period_profile[n]  - j*alpha_e/2
				beta_del_2 = self.beta2[ii] - math.pi/self.period_profile[n]  - j*alpha_e/2

				S_1=[  [j*beta_del_1, 0, 0, 0], [0, j*beta_del_2, 0, 0],
				       [0, 0, -j*beta_del_1, 0],[0, 0, 0, -j*beta_del_2]]

				# S2 = transfert matrix
				S_2=  [[-j*beta_del_1,  0,  -j*kappa_11*np.exp(j*2*beta_del_1*l_0),  -j*kappa_12*np.exp(j*(beta_del_1+beta_del_2)*l_0)],
				       [0,  -j*beta_del_2,  -j*kappa_12*np.exp(j*(beta_del_1+beta_del_2)*l_0),  -j*kappa_22*np.exp(j*2*beta_del_2*l_0)],
				       [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1*l_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1+beta_del_2)*l_0),  j*beta_del_1,  0],
				       [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1+beta_del_2)*l_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2*l_0),  0,  j*beta_del_2]]

				P0=np.matmul(scipy.linalg.expm(np.asarray(S_1)*l_seg),scipy.linalg.expm(np.asarray(S_2)*l_seg))
				if n == 0:
				    P1 = P0*1
				else:
				    P1 = np.matmul(P0,P)
				P = P1

				l_0 += l_seg

			    
			LeftRightTransferMatrix[:,:,ii] = P
			# Calculating In Out Matrix
			# Matrix Switch, flip inputs 1&2 with outputs 1&2
			H = self.switchTop(P)
			InOutTransferMatrix[:,:,ii] = H

			# Calculate Top Down Matrix
			P2 = P
			# switch the order of the inputs/outputs
			P2=np.vstack((P2[3][:], P2[1][:], P2[2][:], P2[0][:])) # switch rows
			P2=self.swap_cols(P2,1,2) # switch columns
			# Matrix Switch, flip inputs 1&2 with outputs 1&2
			P3 = self.switchTop( P2 )
			# switch the order of the inputs/outputs
			P3=np.vstack((P3[3][:], P3[0][:], P3[2][:], P3[1][:])) # switch rows
			P3=self.swap_cols(P3,2,3) # switch columns
			P3=self.swap_cols(P3,1,2) # switch columns

			TopDownTransferMatrix[:,:,ii] = P3
			T[0,ii] = H[0,0]*mode_kappa_a1+H[0,1]*mode_kappa_a2
			R[0,ii] = H[3,0]*mode_kappa_a1+H[3,1]*mode_kappa_a2

			T_co[0,ii] = H[1,0]*mode_kappa_a1+H[1,0]*mode_kappa_a2
			R_co[0,ii] = H[2,0]*mode_kappa_a1+H[2,1]*mode_kappa_a2

			E_Thru[0,ii] = mode_kappa_a1*T[0,ii]+mode_kappa_a2*T_co[0,ii]
			E_Drop[0,ii] = mode_kappa_b1*R_co[0,ii] + mode_kappa_b2*R[0,ii]

		# return results
		self.E_thru = E_Thru
		self.thru = 10*np.log10(np.abs(self.E_thru[0,:])**2)

		self.E_drop = E_Drop
		self.drop = 10*np.log10(np.abs(self.E_drop[0,:])**2)

		self.TransferMatrix = LeftRightTransferMatrix

	def createRandomCDC(self, bounds_kappa = [0, 48e3], bounds_a=[0, 10], bounds_N=[100, 5000], bounds_lambda_B = [1525e-9, 1575e-9], plot=False):

		a = random.uniform(bounds_a[0], bounds_a[-1])
		self.a = np.round(a, 1)
		self.kappa_max = random.uniform(bounds_kappa[0], bounds_kappa[-1])
		self.N = np.random.randint(bounds_N[0], bounds_N[-1])

		lambda_B_0 = random.uniform(bounds_lambda_B[0], bounds_lambda_B[-1])
		segments = np.arange(0,self.N_seg)
		middle = segments[int((self.N_seg+1)/2)]
		lambda_B = (self.central_wvl - lambda_B_0)/ middle * segments + lambda_B_0
		self.period_profile = lambda_B/2.5/2

		if plot:
			plt.plot(lambda_B*1e9,"o")
			plt.title(str((lambda_B_0*1e9)))
			plt.show()


	def simulate(self, bar=True):
		if self.apod_profile is None:
			self.getApodProfile()
		if self.period_profile is None:
			self.getChirpProfile()
		self.getPropConstants(bar)
		self.propagate(bar)


	def writeToFile(self, fileName = "Data/Dataset_v0.txt"):
		self.getPerformance()
		# fileName = "Data/Dataset_v0.txt"
		writeHead = not os.path.exists(fileName)
		with open(fileName,"a") as file:
			header = "a (float), N (int), kappa (float), apodization (1 X 101), period (1 X 101), real(E_drop) (1 X 1001), imag(E_drop) (1 X 1001)"
			if writeHead:
				file.write(header + "\n")

			data = np.array([self.a, self.N, self.kappa_max])
			data = np.append(data, self.apod_profile)
			data = np.append(data, self.period_profile)
			data = np.append(data, np.real(self.E_drop))
			data = np.append(data, np.imag(self.E_drop))
			# data = np.append(data, np.real(self.E_thru))
			# data = np.append(data, np.imag(self.E_thru))
			data = np.reshape(data, (1, -1))		

			np.savetxt(file, data)


	def createDataSet(self, num_samples, bar=True):

		if bar:
			clc()
			print("Création du jeu de données...")
			self.printProgressBar(0, num_samples, prefix = 'Progress:', suffix = 'Complete', length = 50)

		for _ in range(num_samples):
			self.createRandomCDC()
			self.simulate(bar=False)
			self.writeToFile()

			if bar:
				clc()
				print("Creation du jeu de données...")
				self.printProgressBar(_ + 1, num_samples, prefix = 'Progress:', suffix = 'Complete', length = 50)


	def getPerformance(self):
		if self.E_thru is not None:

			# bandwidth and centre wavelength
			dropMax = max(self.drop)
			drop3dB = self.wavelength[self.drop > dropMax - 3]
			ref_wvl = (drop3dB[-1] + drop3dB[0]) /2
			# TODO: something to discard sidelobes from 3-dB bandwidth
			bandwidth = drop3dB[-1] - drop3dB[0]

			# Top flatness assessment
			dropBand = self.drop[self.drop > dropMax - 3]
			avg = np.mean(dropBand)
			std = np.std(dropBand)

			# Extinction ratio
			ER = -1

			# Smoothness
			smoothness = -1

			self.performance = \
				[("Ref. Wvl" , np.round(ref_wvl*1e9,1)           ,  "nm"), \
				("BW"              , np.round(bandwidth*1e9,1)  ,  "nm"), \
				("Max Ref."         , np.round(dropMax,2)          ,  "dB"), \
				("Avg Ref."     , np.round(avg,2)              ,  "dB"), \
				("Std Dev."     , np.round(std,2)              ,  "dB")] \
				# ("Exctinction Ratio"      , np.round(ER,1)               ,  "dB"), \
				# ("Smoothness"             , np.round(smoothness)       ,  " " )]


	# Display Plots and figures of merit 
	def displayResults(self, advanced=False):

		# clc()
		# print("Displaying results.")

		self.getPerformance()


		fig = plt.figure(figsize=(9,6))
		grid = plt.GridSpec(6,3)

		plt.subplot(grid[0:2,0])
		plt.title("Grating Profiles")
		plt.plot(self.apod_profile/1000,".-")
		plt.xticks([])
		plt.ylabel("Coupling (/mm)")
		plt.tick_params(axis='y', direction="in", right=True)
		# plt.text(self.N_seg/2,self.kappa/4/1000,"a = "+str(self.a),ha="center")

		plt.subplot(grid[2:4,0])
		plt.plot(self.period_profile*1e9,".-")
		plt.xticks([])
		plt.ylabel("Pitch (nm)")
		plt.tick_params(axis='y', direction="in", right=True)

		# plt.subplot(grid[4,0])
		# plt.plot(self.w1_profile*1e9,".-",label="wg 1")
		# plt.ylabel("w1 (nm)")
		# plt.tick_params(axis='y', direction="in", right=True)
		# plt.xticks([])

		# plt.subplot(grid[5,0])
		# plt.plot(range(self.N_seg), self.w2_profile*1e9,".-",label="wg 2")
		# plt.xlabel("Segment")
		# plt.ylabel("w2 (nm)")
		# plt.tick_params(axis='y', direction="in", right=True)	


		plt.subplot(grid[0:2,1])
		plt.title("Specifications")
		numElems = 6
		plt.axis([0,1,-numElems+1,1])
		plt.text(0.5,-0,"N : " + str(self.N),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-1,"N_seg : " + str(self.N_seg),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-2,"a : " + str(self.a),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-3,"p: " + str(self.period)+" m",fontsize=11,ha="center",va="bottom")
		# plt.text(0.5,-4,"w1 : " + str(self.w1)+" m",fontsize=11,ha="center",va="bottom")
		# plt.text(0.5,-5,"w2 : " + str(self.w2)+" m",fontsize=11,ha="center",va="bottom")
		plt.xticks([])
		plt.yticks([])
		plt.box(False)


		plt.subplot(grid[0:2,2])
		plt.title("Performance")
		numElems = np.size(self.performance)/3
		plt.axis([0,1,-numElems+1,1])
		for i in np.arange(0,5):
			plt.text(0.5,-i,self.performance[i][0]+" : ",fontsize=11,ha="right",va="bottom")
			plt.text(0.5,-i,str(self.performance[i][1])+" "+self.performance[i][2],fontsize=11,ha="left",va="bottom")
		plt.xticks([])
		plt.yticks([])
		plt.box(False)

		
		plt.subplot(grid[2:,1:])
		plt.plot(self.wavelength*1e9, self.thru, label="Thru port")
		plt.plot(self.wavelength*1e9, self.drop, label="Drop port")
		plt.legend()
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Response (dB)")
		plt.tick_params(axis='y', which='both', labelleft=False, labelright=True, \
						direction="in", right=True)
		plt.tick_params(axis='x', top=True)


		plt.show()







