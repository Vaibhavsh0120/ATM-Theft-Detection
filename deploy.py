from roboflow import Roboflow # type: ignore

# 1. Authenticate with your Roboflow API Key
# 🔑 Replace "YOUR_API_KEY" with your private API key
rf = Roboflow(api_key="cU9XpbWaiDLkNrghIDT0")

# 2. Get your project and version
# 📂 Replace the workspace, project ID, and version number
project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
version = project.version(2) # 👈 Change '1' if your version number is different

# 3. Deploy the trained model
# This is the line you originally had. It will work now that 'version' is defined.
version.deploy(
    model_type="yolov8",
    model_path="D:/D_Downloads/ATM Theft Detection.v1-version-1.yolov8/runs/detect/train",
    filename="weights/best.pt"
)

print("Model deployment to Roboflow initiated successfully!")