import os, time, subprocess
from Utils import commons_utils
import json

###
DIR_S3="/home/ubuntu/s3-drive"
BUCKET_NAME="YOUR_BUCKET_NAME"
AWS_CMD="/usr/local/bin/aws"
###

def send_to_s3(config, dir):
    if is_it_ec2():
        mount_s3_drive()
        s3_path = os.path.join(DIR_S3, dir)
        commons_utils.reset_dir(s3_path)
        command = "cp -R " + dir + "/* " + s3_path
        subprocess.call(command,shell=True)
        # umount_s3()

def mount_s3_drive():
    # if s3fs_running_count() > 0: umount_s3()
    if s3fs_running_count() > 0: return

    command = "s3fs " + BUCKET_NAME + " " + DIR_S3
    subprocess.call(command,shell=True)
    if not s3fs_running_count() > 0: raise Exception("[ERROR] S3 Bucket not mounted locally")

def umount_s3():
    command = "sudo umount " + DIR_S3
    subprocess.call(command,shell=True)
    time.sleep(3)
    if s3fs_running_count() > 0: raise Exception("[ERROR] S3 Bucket is still mounted locally even after umount command")

def s3fs_running_count():
    command = "pgrep s3fs | wc -l"
    return int(subprocess.check_output(command,shell=True))

def is_it_ec2():
    command = "hostname"
    result = str(subprocess.check_output(command, shell=True))
    return "ip-" in result

def shutdown_if_ec2():
    if is_it_ec2():
        command = "sudo poweroff"
        subprocess.call(command, shell=True)

def mark_spot_request_as_cancelled():
    if is_it_ec2():
        spot_id = get_spot_id()
        if spot_id is not None:
            command = AWS_CMD + " ec2 cancel-spot-instance-requests --spot-instance-request-ids " + spot_id
            subprocess.call(command, shell=True)

def get_instance_id():
    if is_it_ec2():
        command = "ec2metadata --instance-id"
        result = str(subprocess.check_output(command, shell=True).decode('utf-8'))
        return result

def get_spot_id():
    if is_it_ec2():
        instance_id = get_instance_id()
        command = AWS_CMD + " ec2 describe-spot-instance-requests --filters 'Name=instance-id,Values=" + instance_id + "'"
        result = str(subprocess.check_output(command, shell=True).decode('utf-8'))
        result = json.loads(result)
        return result['SpotInstanceRequests'][0]['SpotInstanceRequestId']

def get_aws_tag(key):
    if is_it_ec2():
        instance_id = get_instance_id()
        command = AWS_CMD + " ec2 describe-tags --filters 'Name=resource-id,Values=" + instance_id + "' 'Name=key,Values=" + key + "'"
        result = str(subprocess.check_output(command, shell=True).decode('utf-8'))
        result = json.loads(result)
        if len(result['Tags']) > 0:
            return result['Tags'][0]['Value'].split(',')
    return None