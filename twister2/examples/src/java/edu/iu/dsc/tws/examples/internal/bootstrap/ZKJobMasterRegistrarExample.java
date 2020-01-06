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
package edu.iu.dsc.tws.examples.internal.bootstrap;

import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.common.zk.ZKContext;
import edu.iu.dsc.tws.common.zk.ZKJobMasterRegistrar;
import edu.iu.dsc.tws.master.JobMasterContext;

public final class ZKJobMasterRegistrarExample {
  private static final Logger LOG = Logger.getLogger(ZKJobMasterRegistrarExample.class.getName());

  private ZKJobMasterRegistrarExample() {
  }

  /**
   * we assume that we have the Job Master IP address and the port number
   * We will register this pair of information on a ZooKeeper server
   * Workers will discover the Job Master address by querying this ZooKeeper server
   * <p>
   * If there is already a znode on the ZooKeeper with the same name,
   * we delete that znode. It must be from a previous registration session
   * <p>
   * Parameters:
   * the only parameter is the ZooKeeper server address
   * <p>
   * This class is used together with ZKJobMasterFinderExample.java
   * This class registers the Job Master and that class discovers it
   */
  public static void main(String[] args) {

    if (args.length != 1) {
      printUsage();
      return;
    }

    String zkAddress = args[0];
    String jobID = "test-job";
    Config cnfg = buildConfig(zkAddress);

    String jobMasterIP = "x.y.z.t";
    // get the default port
    int jobMasterPort = JobMasterContext.jobMasterPort(cnfg);

    ZKJobMasterRegistrar registrar =
        new ZKJobMasterRegistrar(cnfg, jobMasterIP, jobMasterPort, jobID);
    boolean initialized = registrar.initialize();
    if (!initialized && registrar.sameZNodeExist()) {
      registrar.deleteJobMasterZNode();
      registrar.initialize();
    }

    try {
      long waitDuration = 30;
      LOG.info("Waiting " + waitDuration + "seconds. Will exit afterwards...");
      Thread.sleep(waitDuration * 1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    registrar.close();
    LOG.info("Done, exiting ...");
  }

  /**
   * construct a Config object
   */
  public static Config buildConfig(String zkAddress) {
    return Config.newBuilder()
        .put(ZKContext.SERVER_ADDRESSES, zkAddress)
        .build();
  }

  public static void printUsage() {
    LOG.info("Usage:\n"
        + "java ZKJobMasterRegistrarExample zkAddress");
  }

}
