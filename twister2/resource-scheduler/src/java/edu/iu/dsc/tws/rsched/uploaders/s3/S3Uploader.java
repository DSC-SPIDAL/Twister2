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
package edu.iu.dsc.tws.rsched.uploaders.s3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.scheduler.IUploader;
import edu.iu.dsc.tws.api.scheduler.UploaderException;
import edu.iu.dsc.tws.checkpointing.util.CheckpointingContext;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

public class S3Uploader extends Thread implements IUploader {
  private static final Logger LOG = Logger.getLogger(S3Uploader.class.getName());

  private Config config;
  private String jobID;

  private String localJobPackFile;
  private String s3File;

  // result of uploading
  private boolean uploaded = false;

  @Override
  public void initialize(Config cnfg, String jbID) {
    this.config = cnfg;
    this.jobID = jbID;
  }

  @Override
  public void run() {

    String cmd = String.format("aws s3 cp %s %s", localJobPackFile, s3File);
    LOG.info("cmd for s3 Uploader: " + cmd);
    String[] fullCmd = {"bash", "-c", cmd};

    Process p = null;
    try {
      p = Runtime.getRuntime().exec(fullCmd);
      p.waitFor();
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Exception when executing the command: " + cmd, e);
      return;
    } catch (InterruptedException e) {
      LOG.log(Level.SEVERE, "Exception when waiting the command to complete: " + cmd, e);
      return;
    }

    int exitCode = p.exitValue();

    if (exitCode == 0) {

      uploaded = true;
      LOG.info("Job Package uploaded successfully to: " + s3File);

    } else {
      String failMsg = String.format("Some error occurred when uploading job package %s to s3: %s",
          localJobPackFile, s3File);
      LOG.severe(failMsg);
    }
  }

  @Override
  public URI uploadPackage(String sourceLocation) throws UploaderException {
    localJobPackFile = sourceLocation + "/" + SchedulerContext.jobPackageFileName(config);
    s3File = S3Context.s3BucketName(config) + "/" + JobUtils.createJobPackageFileName(jobID);

    long linkExpDur = S3Context.linkExpirationDuration(config);

    String cmd = String.format("aws s3 presign %s --expires-in %s", s3File, linkExpDur);
    LOG.fine("cmd for s3 URL Generation: " + cmd);
    String[] fullCmd = {"bash", "-c", cmd};

    String url;
    Process p = null;
    try {
      p = Runtime.getRuntime().exec(fullCmd);

      BufferedReader reader = new BufferedReader(
          new InputStreamReader(p.getInputStream()));

      url = reader.readLine();
      p.waitFor();
    } catch (IOException e) {
      throw new UploaderException("Exception when executing the command: " + cmd, e);
    } catch (InterruptedException e) {
      throw new UploaderException("Exception when waiting the script to complete: "
          + cmd, e);
    }

    int exitCode = p.exitValue();

    if (exitCode == 0) {
      LOG.fine("Job Package Download URL: " + url);
      try {
        URI uri = new URI(url);

        // start uploader thread
        start();

        // if this is a resubmitted job, wait for the transfer to complete
        if (CheckpointingContext.startingFromACheckpoint(config)) {
          waitComplete();
        }

        return uri;
      } catch (URISyntaxException e) {
        throw new UploaderException("Can not generate URI for download link: " + url, e);
      }

    } else {
      String failMsg = String.format("Some error occurred when presigning job package %s at s3: %s",
          localJobPackFile, s3File);
      throw new UploaderException(failMsg);
    }
  }

  /**
   * wait for the uploader thread to complete
   */
  private void waitComplete() {
    boolean done = false;
    while (!done) {
      try {
        this.join();
        done = true;
      } catch (InterruptedException e) {
        LOG.log(Level.WARNING, e.getMessage(), e);
        done = false;
      }
    }
  }

  @Override
  public boolean complete() {
    waitComplete();
    return uploaded;
  }

  @Override
  public boolean undo() {

    s3File = S3Context.s3BucketName(config) + "/" + JobUtils.createJobPackageFileName(jobID);

    String cmd = "aws s3 rm " + s3File;
    LOG.fine("cmd for s3 Remover: " + cmd);
    String[] fullCmd = {"bash", "-c", cmd};

    Process p = null;
    try {
      p = Runtime.getRuntime().exec(fullCmd);
      p.waitFor();
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Exception when executing the command: " + cmd, e);
      return false;
    } catch (InterruptedException e) {
      LOG.log(Level.SEVERE, "Exception when waiting the command to complete: " + cmd, e);
      return false;
    }

    int exitCode = p.exitValue();

    if (exitCode == 0) {
      LOG.info("Job Package removed successfully: " + s3File);
      return true;
    } else {
      String failMsg = String.format("Some error occurred when removing the job package %s",
          s3File);
      LOG.severe(failMsg);
      return false;
    }

  }

  @Override
  public void close() {

  }
}
