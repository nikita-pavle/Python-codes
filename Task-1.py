import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

num=[1]
denom=[1,3,5,1]
G=ctrl.TransferFunction(num,denom)
amplitude=int(input("Enter input value : " ))
kp=6
ki=1.5
kd=2.7
reference=amplitude
pid=ctrl.TransferFunction([kd,kp,ki],[1,0])
system=ctrl.feedback(ctrl.series(pid,G))
t,response=ctrl.step_response(amplitude*system,30)
noise = np.random.uniform(low=-0.2, high=0.2, size=len(response))
noisy_response=response+noise
plt.figure()
plt.plot(t,noisy_response)
plt.plot(t,response)
plt.axhline(reference, color="red", linestyle="--", label="Reference")
plt.xlabel("Time")
plt.ylabel("Output")
plt.show()


