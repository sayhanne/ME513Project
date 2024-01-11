import numpy as np
import CESN
import visualizations

if __name__ == '__main__':
    CESN = CESN.CESN()

    context, output = CESN.load_data()
    visualizations.visalize_trajectory(context, output)

    spectral_radius = CESN.normalize_spectral_radius()
    print("Spectral radius :", spectral_radius)
    mse, nrmse = CESN.train(context=context, output=output)
    print('Train Results: ', mse, nrmse)
    # test_mse, test_nrmse = CESN.test(context=context, output=output)
    # print('Test Results :', test_mse, test_nrmse)


