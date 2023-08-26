import torch
import numpy as np
import cv2

'''TartainAir Camera Intrinsics'''
K = np.array([[320 , 0 , 320],
              [0 , 320 , 240],
              [0 ,  0  , 1  ]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_orb_features(images):
  select_kps = 0
  kps = 0
  fails = 0
  images = images.permute(0,2,3,1)
  images = ((images/torch.max(images)).cpu().detach().numpy()*255).astype(np.uint8)
  for i in range(0,len(images),2):
    image = images[i]
    image1 = images[i+1]
    # image = image.permute(1, 2, 0)  # Move the channel dimension to the last dimension
    # image = ((image/torch.max(image)).cpu().detach().numpy()*255).astype(np.uint8)  # Convert to a numpy array of type uint8

    # Convert the image to grayscale if necessary
    # if image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)
    kp2, des2 = sift.detectAndCompute(image1,None)
    pts1 = []
    pts2 = []
    # kp1, des1 = orb.detectAndCompute(image, None)
    # kp2, des2 = orb.detectAndCompute(image1, None)
    try:
      # FLANN parameters
      FLANN_INDEX_KDTREE = 1
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)
      matches = flann.knnMatch(des1,des2,k=2)
      # ratio test as per Lowe's paper
      for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)

      pts1 = np.int32(pts1)
      pts2 = np.int32(pts2)

      F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
      # We select only inlier points
      pts1 = pts1[mask.ravel()==1]
      pts2 = pts2[mask.ravel()==1]
      # print("passed!!")
    except Exception as e:
      #  print(e)
        fails+=1 
        if len(pts1)==0:
          pts1 = [0]

    kps += (len(kp1)+len(kp2))/2
    if type(pts1) is not type(None):
      select_kps+=len(pts1)
    # print(descriptors)
  ret = select_kps
  return torch.tensor([ret],dtype=torch.float32,requires_grad=True).to(device), fails

def sift_matches(image, image1, ret_kps=False):
    select_kp = 0
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)
    kp2, des2 = sift.detectAndCompute(image1,None)
    pts1 = []
    pts2 = []

    kp = 0
    # kp1, des1 = orb.detectAndCompute(image, None)
    # kp2, des2 = orb.detectAndCompute(image1, None)
    try:
      # FLANN parameters
      FLANN_INDEX_KDTREE = 1
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)
      matches = flann.knnMatch(des1,des2,k=2)
      # ratio test as per Lowe's paper
      for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)

      pts1 = np.int32(pts1)
      pts2 = np.int32(pts2)

      F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    #   import ipdb;ipdb.set_trace()
      # We select only inlier points
      pts1 = pts1[mask.ravel()==1]
      pts2 = pts2[mask.ravel()==1] 
      kpts2 = [cv2.KeyPoint(int(x[0]),int(x[1]),5) for x in pts2]
      # print("passed!!")
      kp = (len(kp1)+len(kp2))/2
      if type(pts1) is not type(None):
        select_kp =len(pts1)

    except Exception as e:
        # print(e)
        if len(pts1)==0:
          pts1 = [0]
        kpts2 = []

    if ret_kps:
       return kp, select_kp, kpts2

    return kp, select_kp


def calculate_orb_features_v2(images):
  select_kps = 0
  kps = 0
  fails = 0
  images = images.permute(0,2,3,1)
  device = images.device
  images = ((images/torch.max(images)).cpu().detach().numpy()*255).astype(np.uint8)
  for i in range(0,len(images),2):
    image = images[i]
    image1 = images[i+1]
    kp, select_kp = sift_matches(image,image1)
    kps += kp
    select_kps += select_kp
    if select_kps==0:
      fails +=1
    # print(descriptors)
  if kps == 0:
    ret = 0
  else:
    ret = select_kps/(100*len(images)/2)
  return torch.tensor([ret],dtype=torch.float32,requires_grad=True).to(device), fails
'''
def pose_matches(im,im1):#, prev_kp,prev_desc):
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    # if prev_desc
    kp1, des1 = sift.detectAndCompute(im,None)
    kp2, des2 = sift.detectAndCompute(im1,None)
    pts1 = []
    pts2 = []

    kp = 0
    # kp1, des1 = orb.detectAndCompute(image, None)
    # kp2, des2 = orb.detectAndCompute(image1, None)
    try:
      # FLANN parameters
      FLANN_INDEX_KDTREE = 1
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)
      matches = flann.knnMatch(des1,des2,k=2)
      # ratio test as per Lowe's paper
      for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)

      pts1 = np.int32(pts1)
      pts2 = np.int32(pts2)
      
      # compute relative R,t between ref and cur frame
      E, mask = cv2.findEssentialMat(pts1, pts2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
      _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
      P = np.eye(4)
      P[:3,:3] = R
      P[:3,3:] = t

 
    except Exception as e:
        # print(e)
        P = np.eye(4)
        # if len(pts1)==0:
        #   pts1 = [0]
        # kpts2 = []


    return P
'''
def pose_matches(im,im1):#, prev_kp,prev_desc):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    # if prev_desc
    kp1, des1 = orb.detectAndCompute(im,None)
    kp2, des2 = orb.detectAndCompute(im1,None)
    pts1 = []
    pts2 = []

    kp = 0
    # kp1, des1 = orb.detectAndCompute(image, None)
    # kp2, des2 = orb.detectAndCompute(image1, None)
    try:
      # FLANN parameters
      matcher = cv2.BFMatcher()
      matches = matcher.match(des1,des2)
      # ratio test as per Lowe's paper

      for m in matches[:100]:
          pts2.append(kp2[m.trainIdx].pt)
          pts1.append(kp1[m.queryIdx].pt)
      # for i,(m,n) in enumerate(matches):
      #   if m.distance < 0.8*n.distance:
      #     pts2.append(kp2[m.trainIdx].pt)
      #     pts1.append(kp1[m.queryIdx].pt)

      pts1 = np.int32(pts1)
      pts2 = np.int32(pts2)
      
      # compute relative R,t between ref and cur frame
      E, mask = cv2.findEssentialMat(pts1, pts2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
      _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
      P = np.eye(4)
      P[:3,:3] = R
      P[:3,3:] = t

 
    except Exception as e:
        # print(e)
        P = np.eye(4)
        # if len(pts1)==0:
        #   pts1 = [0]
        # kpts2 = []


    return P

def find_poses(images):
  images = images.permute(0,2,3,1)
  device = images.device
  # images = ((images/torch.max(images)).cpu().detach().numpy()*255).astype(np.uint8)
  images = ((images).cpu().detach().numpy()*255).astype(np.uint8)
  poses = [np.eye(4)]
  for i in range(len(images)-1):
    image = images[i]
    image1 = images[i+1]
    pose = pose_matches(image,image1)
    poses.append(poses[-1]@pose)
  # images = ((images/torch.max(images)).cpu().detach().numpy()*255).astype(np.uint8)

  return torch.tensor(np.array(poses),dtype=torch.float32,requires_grad=True).to(device)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad