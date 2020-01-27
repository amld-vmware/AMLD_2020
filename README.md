## Problem description

As two of the most-mature products, vSphere and vCenter Server provide great deal of customization and flexibility. However, given the complexity of the modern virtualization and the numerous different technologies involved in computing network and storage the customization options for VMware products grew infeasible large for common system administrators to grasp.

At the same time different workloads have different needs. There always is some tradeoff between different functionalities of the system (like throughput and latency or  consolidation and redundancy) and there is not one configuration to serve equally well all kind of workloads. Thus, understanding the purpose of the environment is crucial. 

And while it is easy to profile and optimize single virtual machine, vSphere stack hosts millions virtual environments. In order to approach their needs proactively we have to find a data driven way to classify them.  
</br> 

### The Challenge 

vSphere stack enables (with the explicit agreement of the owners of the environment) the collection of on demand low level performance telemetry. Based on this data we need to identify groups of virtual machines similar with respect to different properties, such as scale, utilization pattern etc. However, we will diverge from the standard clustering algorithms (although we will use one as supporting tool) and try to achieve this through embeddings.  

</br>  

### The Dataset 

The dataset consists of two main data sources related to the performance and the virtual hardware of the virtual machines (VMs).

The **performance telemetry** is organized in a python list, containing multiple python dictionaries. Each dictionary accounts for the data of single VM. The key of the dictionary is the respective ID of the VM, and the value is pandas data frame indexed by the timestamp and containing the actual measurements of each feature.  

</br>

Variable Name |Df index| type|unit | Description|
--- | --- |--- | --- |---
ts  |yes|timestamp|time|Time stamp (yyyy-mm-dd HH:MM:SS) of the observation|
cpu_run  |no|numeric|milliseconds|The time the virtual machine use the CPU|
cpu_ready|no|numeric|milliseconds|The time the virtual machine wants to run a program on the CPU but waited to be scheduled|
mem_active|no|numeric|kiloBytes|The amount of memory actively used by the vm|
mem_activewrite|no|numeric|kiloBytes|Amount of memory actively being written to by the virtual machine.|
net_packetsRx|no|numeric|count|Number of packets received during the interval.|
net_packetsTx|np|numeric|count|Number of packets transmitted during the interval.|

</br> 

The **virtual hardware dataset** is a normal rectangular data frame indexed by the id of the VM. It represents “static” features that account basically for the scale of the system.  

</br>  
  

Variable Name |Df index| type|unit | Description|
--- | --- |--- | --- |---
id  |yes|integer| |Unique identifier of the virtual machine |
memory_mb|no|integer|megabytes|Configured virtual RAM|
num_vcpus|no|integer|count|Number of virtual processor cores|
number_of_nics|no|integer|count|Number of network interface cards|
num_virtual_disks|no|integer|count|Number of the configured hdd|
os_fam|no|categorical|indetity|The operating system of the VM|

</br></br>

### Requirements  
*  Google account
*  Laptop with modern browser

The notebook is precoded and It does not require programming knowledge. You will be encouraged to play around with some of the parameters, that are defined at the beginning of the code chunks.  

### Notebook location 
This repo contains the materials and the data, needed during the workshop. [The notebook, related to the workshop can be accessed from here.](https://drive.google.com/open?id=1mKx2sAdSNuslQtUsRqREsWPiGuBODtEe)  Please save a copy to your personal Drive (Go to **File->Save a copy to Drive** while you are logged in with your Google account).
</br>  
In order to use the notebook, you will need to mount your Drive as a local filesystem (there is a code chunk inside the notebook) and download the dataset from remote data repo. Make sure you have ~ 30Mb of free storage on your Google Drive.  

### Contacts  

**Dragomir Nikolav**  
nikolovd@vmware.com

**Dimira Petrova**  
dpetrova@vmware.com

**Zhivko Kolev**  
zkolev@vmware.com
