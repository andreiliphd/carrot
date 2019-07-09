# Carrot - tensorboard for PyTorch
============

Carrot is a neural network web based analysis tool that allow to see parameters and gradients visually. I want to say that work is in
progress. See Contributing for details.


---

## Features
- You can visually see gradients, parameters, training loss, test loss, training accuracy and test accuracy.

---

## Stack
- PyTorch
- Dash
- MongoDB

---



## Screenshots

![Loss tab](https://github.com/andreiliphd/carrot/blob/master/images/loss_tab.jpg)

![Accuracy tab](https://github.com/andreiliphd/carrot/blob/master/images/accuracy_tab.jpg)

![Parameters tab](https://github.com/andreiliphd/carrot/blob/master/images/parameter_tab.jpg)

![Gradients tab](https://github.com/andreiliphd/carrot/blob/master/images/gradient_tab.jpg)

---

## Setup
Clone this repo:
```
git clone https://github.com/andreiliphd/carrot.git
```
Install all the dependencies.

---


## Usage

- You can use this application to analyze your neural network architecture from inside

---

## Contributing

- Showing multidemenstional plots might require a lot of memory. Although, I solved it partially, improvement might be made.
- MongoDB in implementation is not ideal. Faster to use pickle to save data or even JSON.
- Two preprocessing steps might be removed and heavy workload migh be done on the side of the client not frontend.
- This is a minimum viable product.

---


## License
You can check out the full license in the LICENSE file.

This project is licensed under the terms of the **MIT** license.


