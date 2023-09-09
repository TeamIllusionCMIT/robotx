FROM ros:humble-ros-base-jammy as base

# Set up timezone
ENV TZ=Etc/UTC
RUN echo $TZ > /etc/timezone && \
  ln -fs /usr/share/zoneinfo/$TZ /etc/localtime

# Install required utilities
FROM base as installs
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      gnupg2 \
      lsb-release \
      nano \
      python3-dbg \
      python3-setuptools \
      python3-vcstool \
      ruby \
      sudo \
      wget \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean -qq

# Set up locale
RUN sudo apt-get update && sudo apt-get install locales \
&& sudo locale-gen en_US en_US.UTF-8 \
&& sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ARG ROSDIST=humble
ARG GZDIST=garden
ENV GZ_VERSION garden

from installs as install_tools
# Set up repo to install Ignition
RUN /bin/sh -c 'wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg' \
  && /bin/sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null'


# Install Gazebo (needed for message imports with ros-humble-ros-gzgarden)
RUN apt-get install -y --no-install-recommends \
     gz-${GZDIST} \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean -qq

# Install example ROS package
RUN apt-get install -y --no-install-recommends \
    ros-${ROSDIST}-actuator-msgs \
    ros-${ROSDIST}-ros-gzgarden \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean -qq

from install_tools as runtime
# Copy over script to Docker container
COPY ./run_my_system.bash /

# Use your ros_entrypoint
COPY ./ros_entrypoint.sh /

COPY ./wamv_demo.py /
