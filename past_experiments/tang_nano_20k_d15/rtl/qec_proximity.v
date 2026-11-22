// Distances on the audited lattice with coordinates (round, u, v).
module qec_proximity(
    input wire [11:0] candidate,
    input wire [4:0] t_m2, t_m1, t_z, t_p1, t_p2,
    input wire [4:0] u_m2, u_m1, u_z, u_p1, u_p2,
    input wire [4:0] v_m2, v_m1, v_z, v_p1, v_p2,
    output wire near_one, near_two
);
    wire [4:0] t = {1'b0, candidate[11:8]};
    wire [4:0] u = {1'b0, candidate[7:4]};
    wire [4:0] v = {1'b0, candidate[3:0]};
    wire um2 = u == u_m2, um1 = u == u_m1, uz = u == u_z;
    wire up1 = u == u_p1, up2 = u == u_p2;
    wire vm2 = v == v_m2, vm1 = v == v_m1, vz = v == v_z;
    wire vp1 = v == v_p1, vp2 = v == v_p2;
    wire u_forward = uz | um1, v_forward = vz | vm1;
    wire u_backward = uz | up1, v_backward = vz | vp1;

    wire same_one = (uz & (vz | vm1 | vp1)) | (vz & (um1 | up1));
    wire following_one = u_forward & v_forward;
    wire preceding_one = u_backward & v_backward;
    wire same_two = ((um1 | uz | up1) & (vm1 | vz | vp1)) |
                    (uz & (vm2 | vp2)) | (vz & (um2 | up2));
    wire following_two = (u_forward & (vm2 | v_forward | vp1)) |
                         (v_forward & (um2 | up1));
    wire preceding_two = (u_backward & (vm1 | v_backward | vp2)) |
                         (v_backward & (um1 | up2));
    wire later_two = (um2 | u_forward) & (vm2 | v_forward);
    wire earlier_two = (up2 | u_backward) & (vp2 | v_backward);

    assign near_one = ((t == t_z) & same_one) | ((t == t_p1) & following_one) |
                      ((t == t_m1) & preceding_one);
    assign near_two = ((t == t_z) & same_two) | ((t == t_p1) & following_two) |
                      ((t == t_m1) & preceding_two) | ((t == t_p2) & later_two) |
                      ((t == t_m2) & earlier_two);
endmodule

module qec_three_hop(input wire [11:0] a, b, output wire distance_three);
    wire [11:0] early = a[11:8] <= b[11:8] ? a : b;
    wire [11:0] late = a[11:8] <= b[11:8] ? b : a;
    wire [3:0] dt = late[11:8] - early[11:8];
    function [3:0] excess(input [3:0] x, y, time_gap);
        reg [3:0] backwards;
        begin
            backwards = x - y;
            if (y >= x)
                excess = y - x;
            else
                excess = backwards > time_gap ? backwards - time_gap : 4'd0;
        end
    endfunction
    wire [5:0] length = {2'b0, dt} + {2'b0, excess(early[7:4], late[7:4], dt)} +
                          {2'b0, excess(early[3:0], late[3:0], dt)};
    assign distance_three = length == 6'd3;
endmodule
