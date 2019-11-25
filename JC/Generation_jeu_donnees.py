from Modules import *
from ChirpedContraDC_IAM import ChirpedContraDC


# d = ChirpedContraDC()
# # d.createDataSet(1000)

# d.a = 5
# d.kappa_max = 10e3
# d.N = 1000

# lambda_B_0 = 1575e-9
# segments = np.arange(0,d.N_seg)
# middle = segments[int((d.N_seg+1)/2)]
# lambda_B = (d.central_wvl - lambda_B_0)/ middle * segments + lambda_B_0

# d.period_profile = lambda_B/2.5/2

# d.simulate()
# d.displayResults()
# d.writeToFile(fileName="Data/Dataset_test.txt")

me = np.loadtxt("Data/Dataset_test.txt", skiprows=1)
print(me.shape)

jo = np.loadtxt("Data/ex1.txt", skiprows=1)
jo = np.delete(jo, 3)
print(jo.shape)

print(abs(me - jo))

plt.plot(abs(me - jo))

plt.title("Toi - moi")
plt.show()

