FROM ros:humble-ros-base-jammy as base

ARG ROSDIST=humble
ARG GZDIST=garden
ENV GZ_VERSION garden
ENV ROSDIST ${ROSDIST}
ENV GZDIST ${GZDIST}

# Set up timezone
ENV TZ=Etc/UTC
RUN echo $TZ > /etc/timezone && \
  ln -fs /usr/share/zoneinfo/$TZ /etc/localtime

# Install required utilities
FROM base as installs
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
      gnupg2 \
      lsb-release \
      python3 \
      wget \
      locales

# Set up locale
RUN locale-gen en_US en_US.UTF-8 \
&& update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

FROM installs as install_tools
RUN /bin/sh -c 'wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg' \
  && /bin/sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null'

RUN apt-get update && apt-get install -y --no-install-recommends \
      gz-${GZDIST} \
      ros-${ROSDIST}-actuator-msgs \
      ros-${ROSDIST}-ros-gzgarden \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean -qq


FROM install_tools as runtime
COPY ./run_my_system.bash /
COPY ./ros_entrypoint.sh /
COPY ./wamv_demo.py /

RUN chmod +x /ros_entrypoint.sh
RUN chmod +x /run_my_system.bash
