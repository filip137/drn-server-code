from small_network import mnist_task
import argparse

def main():
    voltage_amp_list = [1, 4, 8]
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    args = p.parse_args()
    for voltage_amp in voltage_amp_list:
        mnist_task()