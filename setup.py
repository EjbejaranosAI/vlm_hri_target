import glob
from setuptools import find_packages, setup

package_name = "vlm_hri_target"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
    ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Edison Bejarano",
    maintainer_email="edison.bejarano@itcl.es",
    description="VLM-based action/social-state recognition for HRI (ROS2 node)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vlm_hri_node = vlm_hri.ros.node:main",
        ],
    },
)
