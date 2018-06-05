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
package edu.iu.dsc.tws.examples.basic;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.common.discovery.IWorkerController;
import edu.iu.dsc.tws.common.discovery.WorkerNetworkInfo;
import edu.iu.dsc.tws.rsched.spi.container.IPersistentVolume;
import edu.iu.dsc.tws.rsched.spi.container.IVolatileVolume;
import edu.iu.dsc.tws.rsched.spi.container.IWorker;
import edu.iu.dsc.tws.rsched.spi.resource.ResourcePlan;

public class BasicK8sWorker implements IWorker {
  private static final Logger LOG = Logger.getLogger(BasicK8sWorker.class.getName());

  @Override
  public void init(Config config,
                   int id,
                   ResourcePlan resourcePlan,
                   IWorkerController workerController,
                   IPersistentVolume persistentVolume,
                   IVolatileVolume volatileVolume) {

    LOG.info("BasicK8sWorker started. Current time: " + System.currentTimeMillis());

    if (volatileVolume != null) {
      volatileVolume.getWorkerDir();
      LOG.info("VolatileVolumeDir: " + volatileVolume.getWorkerDirPath());
    }

    if (persistentVolume != null) {
      // create worker directory
      persistentVolume.getWorkerDir();
    }

    // wait for all workers in this job to join
    List<WorkerNetworkInfo> workerList = workerController.waitForAllWorkersToJoin(10000);
    if (workerList == null) {
      LOG.severe("Can not get all workers to join. Something wrong. .......................");
    } else {
      LOG.info("All workers joined. " + WorkerNetworkInfo.workerListAsString(workerList));
    }

    LOG.info("All workers joined. Current time: " + System.currentTimeMillis());

    echoServer(workerController.getWorkerNetworkInfo());
  }

  /**
   * an echo server.
   * @param workerNetworkInfo
   */
  public static void echoServer(WorkerNetworkInfo workerNetworkInfo) {

    // create socket
    ServerSocket serverSocket = null;
    try {
      serverSocket = new ServerSocket(workerNetworkInfo.getWorkerPort());
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Could not start ServerSocket.", e);
    }

    LOG.info("Echo Started server on port " + workerNetworkInfo.getWorkerPort());

    // repeatedly wait for connections, and process
    while (true) {

      try {
        // a "blocking" call which waits until a connection is requested
        Socket clientSocket = serverSocket.accept();
        LOG.info("Accepted a connection from the client:" + clientSocket.getInetAddress());

        InputStream is = clientSocket.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));

        PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);

        out.println("hello from the server: " + workerNetworkInfo);
        out.println("Will echo your messages:");

        String s;
        while ((s = reader.readLine()) != null) {
          out.println(s);
        }

        // close IO streams, then socket
        LOG.info("Closing the connection with client");
        out.close();
        reader.close();
        clientSocket.close();

      } catch (IOException ioe) {
        throw new IllegalArgumentException(ioe);
      }
    }
  }
}
