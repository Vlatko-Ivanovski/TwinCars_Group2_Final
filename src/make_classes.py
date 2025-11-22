import scipy.io

meta = scipy.io.loadmat("stanford_cars/cars_meta.mat")

class_names = [c[0] for c in meta["class_names"][0]]

with open("data/classes.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"✅ classes.txt created with {len(class_names)} classes")
