import numpy as np

fv= np.array([0.55500015, 0.22262817, 0.46742575, 0.07716229, 0.48056357, 0.73576736,
 0.72758698, 0.29723448, 0.05111963, 0.92799158])
label= -1.0
theta= np.array([0.09009007, 0.22458973, 0.10696886, 0.64798494, 0.10404451, 0.06795626,
 0.06872031, 0.16821736, 0.97809792, 0.0538798])
theta_0= 0.5


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    mul = np.matmul(theta.transpose(), feature_vector)
    s = mul + theta_0
    z = label*s
    l = 1-z
    hinge = np.max([0, l]) #note: Max receieves an array as input.
    return hinge
    raise NotImplementedError

def main():
    print(hinge_loss_single(fv,label,theta,theta_0))  

if __name__ == "__main__":
    main()