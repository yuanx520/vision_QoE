function Q = Kim2012QoS( B, jitter, packetLoss )
    QoS = packetLoss * 10 + jitter * 0.5 + B * 0.01;
    Q = (1-QoS)^(QoS*250/12);
end