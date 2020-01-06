//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package edu.iu.dsc.tws.rsched.uploaders.hdfs;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.TypeUtils;
import edu.iu.dsc.tws.api.scheduler.IUploader;
import edu.iu.dsc.tws.api.scheduler.UploaderException;
import edu.iu.dsc.tws.proto.system.job.JobAPI;


public class HdfsUploader implements IUploader {
  private static final Logger LOG = Logger.getLogger(HdfsUploader.class.getName());
  // get the directory containing the file
  private String destTopologyDirectoryURI;
  private Config config;
  private URI packageURI;

  // The controller on hdfs
  private HdfsController controller;

  @Override
  public void initialize(Config ipconfig, JobAPI.Job job) {
    this.config = ipconfig;

    // Instantiate the hdfs controller
    this.controller = getHdfsController();

    this.destTopologyDirectoryURI = HdfsContext.hdfsTopologiesDirectoryURI(config);
  }

  // Utils method
  protected HdfsController getHdfsController() {
    return new HdfsController(
        HdfsContext.hadoopConfigDirectory(
            config), false); //this second verbose parameter can be read from config file

  }

  // Utils method
  protected boolean isLocalFileExists(String file) {

    return new File(file).isFile();
  }

  @Override
  public URI uploadPackage(String sourceLocation) throws UploaderException {
    // first, check if the topology package exists
    File file = new File(sourceLocation);
    String fileName = file.getName();
    packageURI = TypeUtils.getURI(destTopologyDirectoryURI + "/" + fileName);
    if (!isLocalFileExists(sourceLocation)) {
      throw new UploaderException(
        String.format("Expected topology package file to be uploaded does not exist at '%s'",
            sourceLocation));
    }

    // if the dest directory does not exist, create it.
    if (!controller.exists(destTopologyDirectoryURI)) {
      LOG.info(String.format(
          "The destination directory does not exist. Creating it now at URI '%s'",
          destTopologyDirectoryURI));
      if (!controller.mkdirs(destTopologyDirectoryURI)) {
        throw new UploaderException(
            String.format("Failed to create directory for topology package at URI '%s'",
                destTopologyDirectoryURI));
      }
    } else {
      // if the destination file exists, write a log message
      LOG.info(String.format("Target topology file already exists at '%s'. Overwriting it now",
          packageURI.toString()));
    }

    // copy the topology package to target working directory
    LOG.info(String.format("Uploading topology package at '%s' to target HDFS at '%s'",
        sourceLocation, packageURI.toString()));

    if (!controller.copyFromLocalFile(sourceLocation, packageURI.toString())) {
      throw new UploaderException(
          String.format("Failed to upload the topology package at '%s' to: '%s'",
              sourceLocation, packageURI.toString()));
    }

    try {
      return new URI(destTopologyDirectoryURI + '/' + fileName);
    }  catch (URISyntaxException e) {
      throw new RuntimeException("Invalid file path for topology package destination: "
          + destTopologyDirectoryURI, e);
    }

  }

  @Override
  public boolean undo(Config cnfg, String jobID) {

    return controller.delete(packageURI.toString());
  }

  @Override
  public void close() {
    // Nothing to do here
  }
}
