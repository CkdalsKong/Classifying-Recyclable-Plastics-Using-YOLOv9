from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("./YOLOv9_best.pt")

image_paths = [
    "./vis_ori_image/2023-03-27_200_5.mp4#t=0.jpg",
    "./vis_ori_image/2023-03-27_200_5.mp4#t=1.8.jpg",
    "./vis_ori_image/2024-04-08-14-47-44.bag_86.jpg",
    "./vis_ori_image/20204-04-08-14-35-06.bag_40.jpg",
    "./vis_ori_image/20240404_145512.bag_129.jpg"
]

# Run inference on 'bus.jpg'
results = model(image_paths)  # results list

# Visualize the results
for i, (image_path, result) in enumerate(zip(image_paths, results)):
    # Load the original image
    original_image = Image.open(image_path)
    
    # Plot results image
    im_bgr = result.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    
    # Display the original image and the result side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(im_rgb)
    ax[1].set_title("Result Image")
    ax[1].axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    plt.show()
    
    # Wait for user input to close the plot
    input("Press any key to continue...")
    
    # Save results to disk
    result.save(filename=f"./visualize/results{i}.jpg")